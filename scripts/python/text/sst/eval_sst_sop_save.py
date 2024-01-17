import json
import os
import argparse

import numpy as np
import random
import torch
from torch import nn, optim
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pathlib import Path
from torch.utils.data import DataLoader, Subset
import sys
sys.path.append('lib/exlib/src')
from exlib.modules.sop import SOPConfig, SOPTextCls, get_chained_attr, compress_masks_text, AttributionOutputSOP
# from exlib.modules.sop import SOPImageCls, SOPConfig, get_chained_attr
# from exlib.explainers.archipelago import ArchipelagoImageCls
from exlib.explainers.lime import LimeTextCls
# from exlib.explainers.common import patch_segmenter
from exlib.explainers.shap import ShapTextCls
from exlib.explainers.rise import RiseTextCls
from exlib.explainers.intgrad import IntGradTextCls
# from exlib.explainers import GradCAMImageCls

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from PIL import Image
import jsonlines


from collections import namedtuple

WrappedBackboneOutput = namedtuple("WrappedBackboneOutput", 
                                  ["logits",
                                   "pooler_output"])


class WrappedBackboneModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, inputs=None, **kwargs):
        outputs = self.model(inputs, output_hidden_states=True, **kwargs)
        return WrappedBackboneOutput(outputs.logits, outputs.hidden_states[-1][:,0])
    
class WrappedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, **inputs):
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.logits

class WrappedBackboneModel2(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, inputs=None, **kwargs):
        outputs = self.model(inputs, output_hidden_states=True, **kwargs)
        # import pdb; pdb.set_trace()
        return torch.softmax(outputs.logits, dim=-1)

class SSTDataset(Dataset):
    def __init__(self, data_path, data_size=-1, transform=None):
        self.data_path = data_path
        self.data_size = data_size
        self.transform = transform
        self.documents = []
        self.labels = []
        with jsonlines.open(data_path) as reader:
            for obj in reader:
                self.documents.append(obj['document'])
                self.labels.append(obj['label'])
        self.classes = sorted(set(self.labels))
        
        if data_size != -1:
            # select a subset of the data so that each class has data_size number of documents
            documents = []
            labels = []
            for c in self.classes:
                c_docs = [doc for doc, label in zip(self.documents, self.labels) if label == c]
                documents.extend(c_docs[:data_size])
                labels.extend([c]*data_size)
            self.documents = documents
            self.labels = labels

        assert len(self.documents) == len(self.labels)
        
        print(f'Loaded {len(self.labels)} documents of {len(self.classes)} classes')
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        inputs = self.transform(self.documents[idx])
        inputs['label'] = self.labels[idx]
        return inputs

# EXPLAINER_NAMES = ['lime', 'archipelago', 'rise', 'shap', 'intgrad', 'gradcam']
# EXPLAINER_NAMES = ['lime', 'rise', 'shap', 'intgrad']

if __name__ == '__main__':
    explainer_name = 'sop'
    if len(sys.argv) > 2:
        num_data = int(sys.argv[2])
    else:
        num_data = -1
    # if explainer_name not in EXPLAINER_NAMES:
    #     raise ValueError('Invalid explainer name' + explainer_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SEED = 42
    if SEED != -1:
        # Torch RNG
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        # Python RNG
        np.random.seed(SEED)
        random.seed(SEED)

    # model paths
    backbone_model_name = 'textattack/bert-base-uncased-SST-2'
    backbone_processor_name = 'textattack/bert-base-uncased-SST-2'
    # sop_config_path = 'configs/imagenet_m.json'

    # data paths
    # TRAIN_DATA_DIR = '../data/imagenet_m/train'
    # VAL_DATA_DIR = '../data/imagenet_m/val'

    # training args
    batch_size = 1
    lr = 0.000005
    num_epochs = 20
    warmup_steps = 2000
    mask_batch_size = 4

    # experiment args
    exp_dir = 'exps/sst/best'

    backbone_model = AutoModelForSequenceClassification.from_pretrained(backbone_model_name)
    processor = AutoTokenizer.from_pretrained(backbone_processor_name)
    backbone_config = AutoConfig.from_pretrained(backbone_model_name)

    config = SOPConfig(json_file=os.path.join(exp_dir, 'config.json'),
                    projected_input_scale=2)

    # Path to your dataset file
    train_path = 'data/SST/data/train.jsonl'
    val_path = 'data/SST/data/dev.jsonl'

    # Tokenization function
    def transform(batch):
        return processor(batch, 
                    padding="max_length", 
                    truncation=True, 
                    max_length=512)

    # Load the dataset from the file
    train_size, val_size = -1, -1 #100, 100
    train_dataset = SSTDataset(train_path, data_size=train_size, transform=transform)
    val_dataset = SSTDataset(val_path, data_size=val_size, transform=transform)

    # Create a DataLoader to batch and shuffle the data
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    wrapped_backbone_model = WrappedBackboneModel(backbone_model)
    wrapped_backbone_model = wrapped_backbone_model.to(device)
    class_weights = get_chained_attr(wrapped_backbone_model, config.finetune_layers[0]).weight #.clone().to(device)
    projection_layer = wrapped_backbone_model.model.bert.embeddings.word_embeddings
    # original_model = WrappedModel(backbone_model).to(device)
    # original_model2 = WrappedBackboneModel2(backbone_model).to(device)

    model = SOPTextCls(config, wrapped_backbone_model, class_weights=class_weights, projection_layer=projection_layer)
    state_dict = torch.load(os.path.join(exp_dir, 'checkpoint.pth'))
    model.load_state_dict(state_dict['model'])
    model = model.to(device)
    model.eval();
    
    consistents = 0
    corrects = 0
    total = 0

    os.makedirs(os.path.join(exp_dir, 'attributions', explainer_name), exist_ok=True)

    count = 0

    for batch in tqdm(val_dataloader):
        if not isinstance(batch['input_ids'], torch.Tensor):
            inputs = torch.stack(batch['input_ids']).transpose(0, 1).to(device)
            if 'token_type_ids' in batch:
                token_type_ids = torch.stack(batch['token_type_ids']).transpose(0, 1).to(device)
            else:
                token_type_ids = None
            attention_mask = torch.stack(batch['attention_mask']).transpose(0, 1).to(device)

            # concatenated_rows = [torch.stack(sublist) for sublist in batch['segs']]
            # segs = torch.stack(concatenated_rows).permute(2, 0, 1).to(device).float()
            # print('segs', segs.shape)
        else:
            inputs = batch['input_ids'].to(device)
            if 'token_type_ids' in batch:
                token_type_ids = batch['token_type_ids'].to(device)
            else:
                token_type_ids = None
            attention_mask = batch['attention_mask'].to(device)
            # segs = batch['segs'].to(device).float()
        kwargs = {
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
        }
        inputs_dict = {
            'input_ids': inputs,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        }
        labels = batch['label'].to(device)
        bsz = inputs.shape[0]
        with torch.no_grad():
            # logits = original_model(**inputs_dict)
            expln = model(inputs, kwargs=kwargs, return_tuple=True)
            preds = torch.argmax(expln.logits, dim=-1)
            grouped_attrs_aggr = expln.grouped_attributions.sum(dim=-2)
            aggr_preds = torch.argmax(grouped_attrs_aggr, dim=-1)
            
            connsist = (aggr_preds == preds).sum().item()
            correct = (preds == labels).sum().item()
            
            consistents += connsist
            corrects += correct
            total += bsz

            for j in range(bsz):
                output_filename = f'{count}.pt'
                attributions_results ={
                    'input': inputs[j],
                    'label': labels[j],
                    'logit': expln.logits[j],
                    'expln': expln,
                    'grouped_attrs': expln.grouped_attributions[j],
                    'grouped_attrs_aggr': grouped_attrs_aggr[j],
                    'explns': []
                    
                }
                torch.save(attributions_results, os.path.join(exp_dir, 'attributions', explainer_name, output_filename))
                count += 1
            
            
    print('Consistency: ', consistents / total)
    print('Accuracy: ', corrects / total)
    results = {'consistency': consistents / total, 'accuracy': corrects / total}
