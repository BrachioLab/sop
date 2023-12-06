import json
import os
import argparse

import numpy as np
import random
import torch
from torch import nn, optim
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import get_scheduler

from pathlib import Path
from torch.utils.data import DataLoader, Subset
import sys
sys.path.append('lib/exlib/src')
from exlib.modules.sop import SOPConfig, get_chained_attr

from exlib.modules.sop import SOPTextCls
import copy

from torch.utils.data import DataLoader
from datasets import load_dataset

from collections import namedtuple

import logging


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
backbone_model_name = 'pt_models/multirc_vanilla/best'
backbone_processor_name = 'bert-base-uncased'

# data paths
# TRAIN_DATA_DIR = '../data/imagenet_m/train'
# VAL_DATA_DIR = '../data/imagenet_m/val'

# training args
batch_size = 2
lr = float(sys.argv[1])
# lr = 0.0000005
num_epochs = 20
warmup_steps = 2000
mask_batch_size = 4

# experiment args
exp_dir = f'exps/multirc_{lr}'
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



SENT_SEPS = [processor.convert_tokens_to_ids(processor.tokenize(token)[0]) for token in [';',',','.','?','!',';']]
SEP = processor.convert_tokens_to_ids(processor.tokenize('[SEP]')[0])
print('SEP', SEP, 'SENT_SEPS', SENT_SEPS)

def sent_seg(input_ids):
    segs = []
    count = 1
    for i, input_id in enumerate(input_ids):
        if count in [0, -1]:
            if input_id == SEP:
                count = -1
            segs.append(count)
            continue
        else:
            if input_id in SENT_SEPS:
                segs.append(count)
                count += 1
            elif input_id == SEP:
                if count > 0:
                    count = 0
                    segs.append(count)
                else:
                    segs.append(count)
                    count = -1
            else: # normal character
                segs.append(count)
    return segs

def convert_idx_masks_to_bool_text(masks):
    """
    input: masks (1, seq_len)
    output: masks_bool (num_masks, seq_len)
    """
    unique_idxs = torch.sort(torch.unique(masks)).values
    unique_idxs = unique_idxs[unique_idxs != -1]
    unique_idxs = unique_idxs[unique_idxs != 0]
    idxs = unique_idxs.view(-1, 1)
    broadcasted_masks = masks.expand(unique_idxs.shape[0], 
                                     masks.shape[1])
    masks_bool = (broadcasted_masks == idxs)
    return masks_bool


def get_mask_transform_text(num_masks_max=200, processor=None):
    def mask_transform(mask):
        seg_mask_cut_off = num_masks_max
        # print('mask 1', mask)
        # if mask.max(dim=-1) > seg_mask_cut_off:
        # import pdb; pdb.set_trace()
        if mask.max(dim=-1).values.item() > seg_mask_cut_off:
            mask_new = (mask / (mask.max(dim=-1).values / seg_mask_cut_off)).int().float() + 1
            # bsz, seq_len = mask_new.shape
            # print('mask 2', mask_new)
            # import pdb; pdb.set_trace()
            mask_new[mask == 0] = 0
            mask_new[mask == -1] = -1
            mask = mask_new
        
        if mask.dtype != torch.bool:
            if len(mask.shape) == 1:
                mask = mask.unsqueeze(0)
            # print('mask', mask.shape)
            mask_bool = convert_idx_masks_to_bool_text(mask)
        # print(mask.shape)
        bsz, seq_len = mask.shape
        mask_bool = mask_bool.float()
        
        if bsz < seg_mask_cut_off:
            repeat_count = seg_mask_cut_off // bsz + 1
            mask_bool = torch.cat([mask_bool] * repeat_count, dim=0)

        # add additional mask afterwards
        mask_bool_sum = torch.sum(mask_bool[:seg_mask_cut_off - 1], dim=0, keepdim=True).bool()
        if False in mask_bool_sum:
            mask_bool = mask_bool[:seg_mask_cut_off - 1]
            # import pdb; pdb.set_trace()
            compensation_mask = (1 - mask_bool_sum.int()).bool()
            compensation_mask[mask == 0] = False
            compensation_mask[mask == -1] = False
            mask_bool = torch.cat([mask_bool, compensation_mask])
        else:
            mask_bool = mask_bool[:seg_mask_cut_off]
        return mask_bool
    return mask_transform

mask_transform = get_mask_transform_text(config.num_masks_max)

def transform(batch):
    # Preprocess the image using the ViTImageProcessor
    if processor is not None:
        inputs = processor(batch['passage'], 
                           batch['query_and_answer'], 
                           padding='max_length', 
                           truncation=True, 
                           max_length=512)
        segs = [sent_seg(input_id) for input_id in inputs['input_ids']]
        inputs = {k: torch.tensor(v) for k, v in inputs.items()}
        
        segs_bool = []
        for seg in segs:
            seg_bool = mask_transform(torch.tensor(seg))
            segs_bool.append(seg_bool)
        inputs['segs'] = torch.stack(segs_bool)
        # print("inputs['segs']", inputs['segs'].shape)
        # for k, v in inputs.items():
        #     print(k, v.shape)
        # import pdb; pdb.set_trace()
        return inputs
    else:
        return batch


train_size, val_size = -1, -1

train_dataset = load_dataset('eraser_multi_rc', split='train')
train_dataset = train_dataset.map(transform, batched=True,
                            remove_columns=['passage', 
                                            'query_and_answer',
                                            'evidences'])

val_dataset = load_dataset('eraser_multi_rc', split='validation')
val_dataset = val_dataset.map(transform, batched=True,
                            remove_columns=['passage', 
                                            'query_and_answer',
                                            'evidences'])

if train_size != -1:
    train_dataset = Subset(train_dataset, list(range(train_size)))
if val_size != -1:
    val_dataset = Subset(val_dataset, list(range(val_size)))

# Create a DataLoader to batch and shuffle the data
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



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
projection_layer = wrapped_backbone_model.model.bert.embeddings.word_embeddings

model = SOPTextCls(config, wrapped_backbone_model, class_weights=class_weights, projection_layer=projection_layer)
model = model.to(device)


optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
num_training_steps = len(train_dataloader) * num_epochs
lr_scheduler = get_scheduler(
            'inverse_sqrt',
            optimizer=optimizer, 
            num_warmup_steps=warmup_steps
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
            if not isinstance(batch['input_ids'], torch.Tensor):
                inputs = torch.stack(batch['input_ids']).transpose(0, 1).to(device)
                if 'token_type_ids' in batch:
                    token_type_ids = torch.stack(batch['token_type_ids']).transpose(0, 1).to(device)
                else:
                    token_type_ids = None
                attention_mask = torch.stack(batch['attention_mask']).transpose(0, 1).to(device)

                concatenated_rows = [torch.stack(sublist) for sublist in batch['segs']]
                segs = torch.stack(concatenated_rows).permute(2, 0, 1).to(device).float()
                # print('segs', segs.shape)
            else:
                inputs = batch['input_ids'].to(device)
                if 'token_type_ids' in batch:
                    token_type_ids = batch['token_type_ids'].to(device)
                else:
                    token_type_ids = None
                attention_mask = batch['attention_mask'].to(device)
                segs = batch['segs'].to(device).float()
            kwargs = {
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
            }
            labels = batch['label'].to(device)

            logits = model(inputs, segs=segs, kwargs=kwargs)
            
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
            
            concatenated_rows = [torch.stack(sublist) for sublist in batch['segs']]
            segs = torch.stack(concatenated_rows).permute(2, 0, 1).to(device).float()
            # print('segs', segs.shape)
        else:
            inputs = batch['input_ids'].to(device)
            if 'token_type_ids' in batch:
                token_type_ids = batch['token_type_ids'].to(device)
            else:
                token_type_ids = None
            attention_mask = batch['attention_mask'].to(device)
            segs = batch['segs'].to(device).float()
        kwargs = {
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
        }
        labels = batch['label'].to(device)
            
        
        optimizer.zero_grad()
        logits = model(inputs, segs=segs, mask_batch_size=mask_batch_size, kwargs=kwargs)
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