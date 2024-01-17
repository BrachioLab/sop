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
from exlib.modules.sop import SOPImageCls, SOPConfig, get_chained_attr

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
    

if __name__ == '__main__':
    explainer_name = 'sop'

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
    backbone_model_name = 'pt_models/vit-base-patch16-224'
    backbone_processor_name = 'google/vit-base-patch16-224'
    # sop_config_path = 'configs/imagenet_m.json'

    # data paths
    TRAIN_DATA_DIR = 'data/imagenet/train'
    VAL_DATA_DIR = 'data/imagenet/val'

    # training args
    batch_size = 1
    lr = 0.000005
    num_epochs = 20
    warmup_steps = 2000
    mask_batch_size = 64

    # experiment args
    exp_dir = 'exps/imagenet_2h_10/best'

    backbone_model = AutoModelForImageClassification.from_pretrained(backbone_model_name)
    processor = AutoImageProcessor.from_pretrained(backbone_processor_name)
    backbone_config = AutoConfig.from_pretrained(backbone_model_name)

    config = SOPConfig()
    config.update_from_json(os.path.join(exp_dir, 'config.json'))

    import torchvision.transforms as transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    def transform(image):
        # Preprocess the image using the ViTImageProcessor
        image = image.convert("RGB")
        inputs = processor(image, return_tensors='pt')
        return inputs['pixel_values'].squeeze(0)

    # Load the dataset
    # train_dataset = ImageFolder(root=TRAIN_DATA_DIR, transform=transform)
    val_dataset = ImageFolder(root=VAL_DATA_DIR, transform=transform)

    # Use subset for testing purpose
    # num_data = 2
    # train_dataset = Subset(train_dataset, range(num_data))
    # val_dataset = Subset(val_dataset, range(num_data))

    # Create a DataLoader to batch and shuffle the data
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    wrapped_backbone_model = WrappedBackboneModel(backbone_model)
    wrapped_backbone_model = wrapped_backbone_model.to(device)
    class_weights = get_chained_attr(wrapped_backbone_model, config.finetune_layers[0]).weight #.clone().to(device)

    model = SOPImageCls(config, wrapped_backbone_model, class_weights=class_weights, projection_layer=None)
    state_dict = torch.load(os.path.join(exp_dir, 'checkpoint.pth'))
    model.load_state_dict(state_dict['model'])
    model = model.to(device)
    model.eval();

    consistents = 0
    corrects = 0
    total = 0

    for batch in tqdm(val_dataloader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        bsz = inputs.shape[0]
        with torch.no_grad():
            expln = model(inputs, return_tuple=True)
            preds = torch.argmax(expln.logits, dim=-1)
            grouped_attrs_aggr = expln.grouped_attributions.sum(dim=-2)
            aggr_preds = torch.argmax(grouped_attrs_aggr, dim=-1)
            
            connsist = (aggr_preds == preds).sum().item()
            correct = (preds == labels).sum().item()
            
            consistents += connsist
            corrects += correct
            total += bsz
            
    print('Consistency: ', consistents / total)
    print('Accuracy: ', corrects / total)
    results = {'consistency': consistents / total, 'accuracy': corrects / total}

    os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
    output_file = os.path.join(exp_dir, 'results', explainer_name + '_consistency.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)