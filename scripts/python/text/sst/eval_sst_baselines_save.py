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
from exlib.explainers.archipelago import ArchipelagoTextCls
from exlib.explainers.idg import IDGTextCls
from exlib.explainers.pls import PLSTextCls

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
    
    def forward(self, inputs):
        outputs = self.model(inputs, output_hidden_states=True)
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

class WrappedBackboneModel3(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, inputs=None, **kwargs):
        # device = next(self.model.parameters()).device
        # inputs = inputs.to(device)
        # kwargs = {k: v.to(device) for k, v in kwargs.items()}
        outputs = self.model(inputs, output_hidden_states=True, **kwargs)
        # import pdb; pdb.set_trace()
        return outputs.logits

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

EXPLAINER_NAMES = ['lime', 'archipelago', 'rise', 'shap', 'intgrad', 'idg', 'pls']
# EXPLAINER_NAMES = ['lime', 'rise', 'shap', 'intgrad']

if __name__ == '__main__':
    explainer_name = sys.argv[1]
    if len(sys.argv) > 2:
        num_data = int(sys.argv[2])
    else:
        num_data = -1
    if explainer_name not in EXPLAINER_NAMES:
        raise ValueError('Invalid explainer name' + explainer_name)

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
    # exp_dir = 'exps/sst_m_1h_gg2.0_gs1.0/best'
    exp_dir = 'exps/sst_m_1h_gg2.0_gs1.0_ggta1.0_ggtb0.1/best'

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
    original_model = WrappedModel(backbone_model).to(device)
    original_model2 = WrappedBackboneModel2(backbone_model).to(device)
    original_model3 = WrappedBackboneModel3(backbone_model).to(device)

    model = SOPTextCls(config, wrapped_backbone_model, class_weights=class_weights, projection_layer=projection_layer)
    state_dict = torch.load(os.path.join(exp_dir, 'checkpoint.pth'))
    model.load_state_dict(state_dict['model'])
    model = model.to(device)
    model.eval();
       

    if explainer_name == 'lime':
        eik = {
            "top_labels": 2, 
            "num_samples": 500
        }

        def split_expression(x):
            tokens = x.split()
            return tokens
            
        ltek = {
            "mask_string": "[MASK]",
            "split_expression": split_expression
        }

        explainer = LimeTextCls(original_model, processor,
                                LimeTextExplainerKwargs=ltek,
                                explain_instance_kwargs=eik).to(device)
    elif explainer_name == 'shap':
        explainer = ShapTextCls(original_model, processor).to(device)
    elif explainer_name == 'rise':
        explainer = RiseTextCls(original_model2).to(device)
    elif explainer_name == 'intgrad':
        explainer = IntGradTextCls(original_model2, projection_layer=projection_layer).to(device)
    elif explainer_name == 'archipelago':
        explainer = ArchipelagoTextCls(backbone_model).to(device)
    elif explainer_name == 'idg':
        explainer = IDGTextCls(original_model3, processor).to(device)
    elif explainer_name == 'pls':
        explainer = PLSTextCls(backbone_model, processor).to(device)
    else:
        raise ValueError('Invalid explainer name' + explainer_name)
    
    explainer = explainer.to(device)
        
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

        else:
            inputs = batch['input_ids'].to(device)
            if 'token_type_ids' in batch:
                token_type_ids = batch['token_type_ids'].to(device)
            else:
                token_type_ids = None
            attention_mask = batch['attention_mask'].to(device)

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
            logits = original_model(**inputs_dict)
            preds = torch.argmax(logits, dim=-1)

            if explainer_name in ['lime']:
                expln = explainer(inputs, labels)
            elif explainer_name in ['shap', 'idg']:
                inputs_raw = [processor.decode(input_ids_i).replace('[CLS]', '').replace('[PAD]', '').strip() 
                            for input_ids_i in inputs]

                expln = explainer(inputs_raw, labels)
            elif explainer_name in ['rise']:
                expln = explainer(inputs, labels, kwargs=kwargs)
            elif explainer_name in ['intgrad']:
                expln = explainer(inputs, labels, x_kwargs=kwargs)
            elif explainer_name in ['archipelago', 'pls']:
                expln = explainer(inputs, labels, **kwargs)
            else:
                raise ValueError('Invalid explainer name' + explainer_name)

            aggr_preds = []
            explns = []
            for j in range(bsz):
                if explainer_name == 'lime':
                    grouped_attrs = []
                    for i in tqdm(range(2)):
                        grouped_attrs.append([v for k, v in expln.explainer_output[j].local_exp[i]])
                    grouped_attrs_aggr = torch.tensor([sum(ga) for ga in grouped_attrs]).to(device)
                    aggr_pred = torch.argmax(grouped_attrs_aggr)
                elif explainer_name == 'shap':
                    grouped_attrs = torch.tensor(expln.explainer_output.values[j]).to(device)
                    grouped_attrs_aggr = grouped_attrs.sum(0)
                    aggr_pred = torch.argmax(grouped_attrs_aggr)
                elif explainer_name == 'rise':
                    grouped_attrs = expln.explainer_output[j]
                    grouped_attrs_aggr = expln.explainer_output[j].sum(-1)
                    aggr_pred = torch.argmax(grouped_attrs_aggr)
                elif explainer_name == 'intgrad':
                    grouped_attrs = []
                    for i in tqdm(range(2)):
                        intgrad_expln = explainer(inputs[j][None], torch.tensor([i]).to(device))
                        grouped_attrs.append(intgrad_expln.attributions.view(-1))
                    grouped_attrs = torch.cat(grouped_attrs)
                    grouped_attrs_aggr = grouped_attrs.sum(-1)
                    aggr_pred = torch.argmax(grouped_attrs_aggr)
                elif explainer_name == 'archipelago':
                    grouped_attrs = []
                    for i in tqdm(range(2)):
                        expln_i = explainer(inputs, torch.tensor([i]), **kwargs)
                        grouped_attrs.append(expln_i.explainer_output['mask_weights'][j])
                    grouped_attrs_aggr = torch.tensor([sum(ga) for ga in grouped_attrs]).to(device)
                    aggr_pred = torch.argmax(grouped_attrs_aggr)
                elif explainer_name == 'idg':
                    grouped_attrs = []
                    for i in tqdm(range(2)):
                        idg_expln = explainer(inputs_raw, torch.tensor([i]))
                        grouped_attrs.append(expln.attributions.view(-1))
                    grouped_attrs_aggr = torch.stack(grouped_attrs).sum(-1).to(device)
                    aggr_pred = torch.argmax(grouped_attrs_aggr)
                elif explainer_name == 'pls':
                    grouped_attrs = []
                    for i in tqdm(range(2)):
                        expln_i = explainer(inputs, torch.tensor([i]).to(device), **kwargs)
                        grouped_attrs.append(expln_i.explainer_output['mask_weights'])
                    grouped_attrs_aggr = torch.tensor(grouped_attrs).to(device) #torch.tensor([sum(ga) for ga in grouped_attrs]).to(device)
                    aggr_pred = torch.argmax(grouped_attrs_aggr)
                else:
                    raise ValueError('Invalid explainer name' + explainer_name)
                aggr_preds.append(aggr_pred)

                output_filename = f'{count}.pt'
                attributions_results = {
                    'input': inputs[j],
                    'token_type_ids': token_type_ids[j],
                    'attention_mask': attention_mask[j],
                    'label': labels[j],
                    'logit': logits[j],
                    'expln': expln,
                    'grouped_attrs': grouped_attrs,
                    'grouped_attrs_aggr': grouped_attrs_aggr,
                    'explns': explns
                }
                torch.save(attributions_results, os.path.join(exp_dir, 'attributions', explainer_name, output_filename))
                count += 1

            aggr_preds = torch.stack(aggr_preds)
            print('aggr_preds', aggr_preds.device, 'preds', preds.device, 'labels', labels.device)
            connsist = (aggr_preds == preds).sum().item()
            correct = (preds == labels).sum().item()

            consistents += connsist
            corrects += correct
            total += bsz
            
    print('Consistency: ', consistents / total)
    print('Accuracy: ', corrects / total)
    results = {'consistency': consistents / total, 'accuracy': corrects / total}
