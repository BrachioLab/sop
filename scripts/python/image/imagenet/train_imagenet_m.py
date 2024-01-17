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
from exlib.modules.sop import SOPImageCls, SOPConfig, get_chained_attr, get_inverse_sqrt_with_separate_heads_schedule_with_warmup

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
backbone_model_name = 'pt_models/vit-base-patch16-224-imagenet10cls'
backbone_processor_name = 'google/vit-base-patch16-224'
# sop_config_path = 'configs/imagenet_m.json'

# data paths
TRAIN_DATA_DIR = 'data/imagenet_m/train'
VAL_DATA_DIR = 'data/imagenet_m/val'

# training args
batch_size = 16
lr = 0.000005
num_epochs = 20
warmup_steps = 2000
mask_batch_size = 64

# experiment args
exp_dir = 'exps/imagenet_m_2h'
os.makedirs(exp_dir, exist_ok=True)

backbone_model = AutoModelForImageClassification.from_pretrained(backbone_model_name)
processor = AutoImageProcessor.from_pretrained(backbone_processor_name)
backbone_config = AutoConfig.from_pretrained(backbone_model_name)

config = SOPConfig(
    attn_patch_size=16,
    num_heads=2,
    num_masks_sample=20,
    num_masks_max=200,
    finetune_layers=['model.classifier']
)
config.__dict__.update(backbone_config.__dict__)
config.num_labels = len(backbone_config.label2id)
# config.save_pretrained(exp_dir)

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def transform(image):
    # Preprocess the image using the ViTImageProcessor
    image = image.convert("RGB")
    inputs = processor(image, return_tensors='pt')
    return inputs['pixel_values'].squeeze(0)

# Load the dataset
train_dataset = ImageFolder(root=TRAIN_DATA_DIR, transform=transform)
val_dataset = ImageFolder(root=VAL_DATA_DIR, transform=transform)

# Use subset for testing purpose
# num_data = 100
# train_dataset = Subset(train_dataset, range(num_data))
# val_dataset = Subset(val_dataset, range(num_data))

# Create a DataLoader to batch and shuffle the data
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
    
wrapped_backbone_model = WrappedBackboneModel(backbone_model)
wrapped_backbone_model = wrapped_backbone_model.to(device)
class_weights = get_chained_attr(wrapped_backbone_model, config.finetune_layers[0]).weight #.clone().to(device)

model = SOPImageCls(config, wrapped_backbone_model, class_weights=class_weights)
model = model.to(device)

from transformers import get_scheduler

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
num_training_steps = len(train_dataloader) * num_epochs
train_rep_step_size = int(num_training_steps / config.num_heads)
lr_scheduler = get_inverse_sqrt_with_separate_heads_schedule_with_warmup(
            optimizer=optimizer, 
            num_warmup_steps=warmup_steps,
            num_steps_per_epoch=train_rep_step_size,
            num_heads=config.num_heads
        )
criterion = nn.CrossEntropyLoss()

def eval(model, dataloader, criterion):
    print('Eval ...')
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar_eval = tqdm(range(len(dataloader)))
        for i, batch in enumerate(dataloader):
            # Now you can use `inputs` and `labels` in your training loop.
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            logits = model(inputs)
            
            # val loss
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # acc
            _, predicted = torch.max(logits.data, 1)
            correct += (predicted == labels).sum().item()
            
            total += labels.size(0)
            
            progress_bar_eval.update(1)
    
    val_acc = correct / total
    val_loss = total_loss / total
    
    model.train()
    
    return {
        'val_acc': val_acc,
        'val_loss': val_loss
    }

import logging

track = True
# track = False

if track:
    import wandb
    wandb.init(project='sop')
    wandb.run.name = os.path.basename(exp_dir)

# Iterate over the data
best_val_acc = 0.0
step = 0
train_log_interval = 100
val_eval_interval = 1000

logging.basicConfig(filename=os.path.join(exp_dir, 'train.log'), level=logging.INFO)

progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    running_loss = 0.0
    running_total = 0
    for i, batch in enumerate(train_dataloader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        train_rep_step = step // train_rep_step_size
        logits = model(inputs, epoch=train_rep_step, mask_batch_size=mask_batch_size)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * labels.size(0)
        running_total += labels.size(0)
        
        if i % train_log_interval == train_log_interval - 1 or i == len(train_dataloader) - 1:
            # Print training loss every 100 batches
            curr_lr = float(optimizer.param_groups[0]['lr'])
            log_message = f'Epoch {epoch}, Batch {i + 1}, Loss {running_loss / running_total:.4f}, LR {curr_lr:.8f}'
            print(log_message)
            logging.info(log_message)
            if track:
                wandb.log({'train_loss': running_loss / running_total,
                        'lr': curr_lr,
                        'epoch': epoch,
                        'step': step})
            running_loss = 0.0
            running_total = 0
            
        if i % val_eval_interval == val_eval_interval - 1 or i == len(train_dataloader) - 1:
            val_results = eval(model, val_dataloader, criterion)
            val_acc = val_results['val_acc']
            val_loss = val_results['val_loss']
            log_message = f'Epoch {epoch}, Step {step}, Val acc {val_acc:.4f}, Val loss {val_loss:.4f}'
            print(log_message)
            logging.info(log_message)
            if track:
                wandb.log({'val_acc': val_acc,
                           'val_loss': val_loss,
                        'epoch': epoch,
                        'step': step})
            
            last_dir = os.path.join(exp_dir, 'last')
            best_dir = os.path.join(exp_dir, 'best')
            checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'step': step,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs(best_dir, exist_ok=True)
                best_checkpoint_path = os.path.join(best_dir, 'checkpoint.pth')
                torch.save(checkpoint, best_checkpoint_path)
                config_best_checkpoint_path = os.path.join(best_dir, 'config.json')
                config.save_to_json(config_best_checkpoint_path)
                print(f'Best checkpoint saved at {best_checkpoint_path}')
                
                # model.save_pretrained(best_dir)
            # model.save_pretrained(last_dir)
            os.makedirs(last_dir, exist_ok=True)
            last_checkpoint_path = os.path.join(last_dir, 'checkpoint.pth')
            torch.save(checkpoint, last_checkpoint_path)
            config_last_checkpoint_path = os.path.join(last_dir, 'config.json')
            config.save_to_json(config_best_checkpoint_path)
            print(f'Last checkpoint saved at {last_checkpoint_path}')
            
        lr_scheduler.step()
        progress_bar.update(1)
        
        step += 1
        
model.save(exp_dir)