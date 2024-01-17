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
from exlib.modules.sop import SOPConfig, get_chained_attr, SOPTextCls

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
backbone_model_name = 'joeddav/distilbert-base-uncased-agnews-student'
backbone_processor_name = 'joeddav/distilbert-base-uncased-agnews-student'

# training args
batch_size = 16
lr = 0.0000005
num_epochs = 20
warmup_steps = 2000
mask_batch_size = 4

# experiment args
exp_dir = 'exps/agnews'
os.makedirs(exp_dir, exist_ok=True)

backbone_model = AutoModelForSequenceClassification.from_pretrained(backbone_model_name)
processor = AutoTokenizer.from_pretrained(backbone_processor_name)
backbone_config = AutoConfig.from_pretrained(backbone_model_name)


config = SOPConfig(
    # attn_patch_size=16,
    num_heads=1,
    num_masks_sample=8,
    num_masks_max=16,
    finetune_layers=['model.classifier']
)
config.__dict__.update(backbone_config.__dict__)
config.num_labels = len(backbone_config.label2id)

# config.save_pretrained(exp_dir)

from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import jsonlines


class AgNewsDataset(Dataset):
    def __init__(self, data_path, label2id, id2label, data_size=-1, transform=None):
        self.label_config_map = {
            'Business': 'business', 
            'Sci/Tech': 'science/tech', 
            'Sports': 'sports', 
            'World': 'the world'
        }
        self.data_path = data_path
        self.data_size = data_size
        self.transform = transform
        self.label2id = label2id
        self.id2label = id2label
        self.documents = []
        self.labels = []
        with jsonlines.open(data_path) as reader:
            for obj in reader:
                self.documents.append(obj['document'])
                self.labels.append(self.label2id[self.label_config_map[obj['label']]])  #self.label2id[obj['label'].lower()])
        self.classes = sorted(set(self.labels))
        print(self.classes)
        
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

        
# Path to your dataset file
train_path = 'data/agnews/data/train.jsonl'
val_path = 'data/agnews/data/dev.jsonl'

# Tokenization function
def transform(batch):
    return processor(batch, 
                   padding="max_length", 
                   truncation=True, 
                   max_length=512)

# Load the dataset from the file
# train_size, val_size = -1, -1
train_size, val_size = 100, 100
train_dataset = AgNewsDataset(train_path, config.label2id, config.id2label, data_size=train_size, transform=transform)
val_dataset = AgNewsDataset(val_path, config.label2id, config.id2label, data_size=val_size, transform=transform)

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
    
    def forward(self, inputs=None, **kwargs):
        outputs = self.model(inputs, output_hidden_states=True, **kwargs)
        return WrappedBackboneOutput(outputs.logits, outputs.hidden_states[-1][:,0])
    
wrapped_backbone_model = WrappedBackboneModel(backbone_model)
wrapped_backbone_model = wrapped_backbone_model.to(device)
class_weights = get_chained_attr(wrapped_backbone_model, config.finetune_layers[0]).weight #.clone().to(device)
print(wrapped_backbone_model.model)
projection_layer = wrapped_backbone_model.model.distilbert.embeddings.word_embeddings

model = SOPTextCls(config, wrapped_backbone_model, class_weights=class_weights, projection_layer=projection_layer)
model = model.to(device)

from transformers import get_scheduler

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
num_training_steps = len(train_dataloader) * num_epochs
lr_scheduler = get_scheduler(
            'inverse_sqrt',
            optimizer=optimizer, 
            num_warmup_steps=warmup_steps
        )
criterion = nn.CrossEntropyLoss()

def eval(model, dataloader, criterion, sop=True):
    print('Eval ...')
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar_eval = tqdm(range(len(dataloader)))
        for i, batch in enumerate(dataloader):
            # Now you can use `inputs` and `labels` in your training loop.
            if not isinstance(batch['input_ids'], torch.Tensor):
                inputs = torch.stack(batch['input_ids']).transpose(0, 1).to(device)
                if 'token_type_ids' in batch:
                    token_type_ids = torch.stack(batch['token_type_ids']).transpose(0, 1).to(device)
                else:
                    token_type_ids = None
                attention_mask = torch.stack(batch['attention_mask']).transpose(0, 1).to(device)

                # print('segs', segs.shape)
            else:
                inputs = batch['input_ids'].to(device)
                if 'token_type_ids' in batch:
                    token_type_ids = batch['token_type_ids'].to(device)
                else:
                    token_type_ids = None
                attention_mask = batch['attention_mask'].to(device)
            kwargs = {
                # 'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
            }
            labels = batch['label'].to(device)
            # import pdb; pdb.set_trace()

            if sop:
                logits = model(inputs, kwargs=kwargs)
            else:
                logits = model(inputs, **kwargs).logits
            
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

backbone_val_results = eval(wrapped_backbone_model, val_dataloader, criterion, sop=False)
backbone_val_acc = backbone_val_results['val_acc']
backbone_val_acc

import logging

# track = True
track = False

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

model.train()

progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    running_loss = 0.0
    running_total = 0
    for i, batch in enumerate(train_dataloader):
        # import pdb; pdb.set_trace()
        # inputs, labels = batch
        # inputs, labels = inputs.to(device), labels.to(device)
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
            # 'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
        }
        labels = batch['label'].to(device)
            
        
        optimizer.zero_grad()
        logits = model(inputs, mask_batch_size=mask_batch_size, kwargs=kwargs)
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