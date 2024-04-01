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
from exlib.modules.sop import SOPConfig, get_chained_attr, SOPTextCls, get_inverse_sqrt_with_separate_heads_schedule_with_warmup
from exlib.modules.fresh import FRESH

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


def parse_args():
    parser = argparse.ArgumentParser()

    # paths and info
    parser.add_argument('--group-gen-scale', type=float, 
                        default=1,
                        help='group gen scale')
    parser.add_argument('--group-sel-scale', type=float, 
                        default=1,
                        help='group sel scale')
    parser.add_argument('--group-gen-temp-alpha', type=float, 
                        default=1,
                        help='group gen temp alpha (outside log)')
    parser.add_argument('--group-gen-temp-beta', type=float, 
                        default=1,
                        help='group gen temp beta (inside log)')
    parser.add_argument('--num-heads', type=int, 
                        default=1,
                        help='num heads')
    parser.add_argument('--model', type=str, 
                        default='sop', choices=['sop', 'fresh'],
                        help='num heads')
    parser.add_argument('--reload-checkpoint', 
                        default=False, action='store_true',
                        help='if true, then reload checkpoint')
    parser.add_argument('--scheduler-type', type=str,
                        default='inv_sqrt', choices=['inv_sqrt', 'cosine'],
                        help='scheduler type')
    parser.add_argument('--lr', type=float, 
                        default=0.000005,
                        help='lr')
    parser.add_argument('--num-epochs', type=int,
                        default=20,
                        help='num epochs')
    
    return parser

parser = parse_args()
args = parser.parse_args()

GROUP_GEN_SCALE = args.group_gen_scale
GROUP_SEL_SCALE = args.group_sel_scale
GROUP_GEN_TEMP_ALPHA = args.group_gen_temp_alpha
GROUP_GEN_TEMP_BETA = args.group_gen_temp_beta
NUM_HEADS = args.num_heads

# model paths
backbone_model_name = 'pt_models/multirc_vanilla/best'
backbone_processor_name = 'bert-base-uncased'

# training args
batch_size = 2
lr = args.lr #0.0000005
num_epochs = args.num_epochs
warmup_steps = 2000
mask_batch_size = 4
group_gen_scale = GROUP_GEN_SCALE
group_sel_scale = GROUP_SEL_SCALE
group_gen_temp_alpha = GROUP_GEN_TEMP_ALPHA
group_gen_temp_beta = GROUP_GEN_TEMP_BETA

exp_name = 'multirc_{}h_gg{}_gs{}_ggta{}_ggtb{}_ep{}_lr{}_sc{}'.format(NUM_HEADS, group_gen_scale, group_sel_scale, 
                group_gen_temp_alpha, group_gen_temp_beta, num_epochs, lr, args.scheduler_type)

if args.model == 'fresh':
    exp_name += '_fresh'

# experiment args
exp_dir = f'exps/{exp_name}'
os.makedirs(exp_dir, exist_ok=True)

last_dir = os.path.join(exp_dir, 'last')
best_dir = os.path.join(exp_dir, 'best')
best_checkpoint_path = os.path.join(best_dir, 'checkpoint.pth')
last_checkpoint_path = os.path.join(last_dir, 'checkpoint.pth')
config_best_checkpoint_path = os.path.join(best_dir, 'config.json')
config_last_checkpoint_path = os.path.join(last_dir, 'config.json')

backbone_model = AutoModelForSequenceClassification.from_pretrained(backbone_model_name)
processor = AutoTokenizer.from_pretrained(backbone_processor_name)
backbone_config = AutoConfig.from_pretrained(backbone_model_name)

config = SOPConfig(
    # attn_patch_size=16,
    num_heads=NUM_HEADS,
    num_masks_sample=8,
    num_masks_max=16,
    finetune_layers=['model.classifier'],
    group_gen_scale=group_gen_scale,
    group_sel_scale=group_sel_scale,
    group_gen_temp_alpha=group_gen_temp_alpha,
    group_gen_temp_beta=group_gen_temp_beta,
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

from datasets import load_dataset

train_size, val_size = -1, -1
# train_size, val_size = 100, 100

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
projection_layer = wrapped_backbone_model.model.bert.embeddings.word_embeddings

if args.model == 'sop':
    model = SOPTextCls(config, wrapped_backbone_model, class_weights=class_weights, projection_layer=projection_layer)
else:
    import copy
    fresh_config = copy.deepcopy(backbone_config)
    fresh_config.__dict__.update(config.__dict__)
    fresh_config.finetune_layers = [fresh_config.finetune_layers[0].replace('model.', '')]
    model = FRESH(fresh_config,
                    backbone_model,
                    model_type='text',
                    return_tuple=True,
                    postprocess_attn=lambda x: x.attentions[-1].mean(dim=1)[:,0],
                    postprocess_logits=lambda x: x.logits)
model = model.to(device)

from transformers import get_scheduler

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
num_training_steps = len(train_dataloader) * num_epochs
train_rep_step_size = int(num_training_steps / config.num_heads)
if args.scheduler_type == 'inv_sqrt':
    lr_scheduler = get_inverse_sqrt_with_separate_heads_schedule_with_warmup(
                optimizer=optimizer, 
                num_warmup_steps=warmup_steps,
                num_steps_per_epoch=train_rep_step_size,
                num_heads=config.num_heads
            )
else:
    lr_scheduler = get_scheduler(
        'cosine',
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
criterion = nn.CrossEntropyLoss()

if args.reload_checkpoint and os.path.exists(last_checkpoint_path):
    checkpoint = torch.load(last_checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    checkpoint_step = checkpoint['step']
    checkpoint_epoch = checkpoint['epoch']
    print(f'Loaded checkpoint from {last_checkpoint_path} at step {checkpoint_step} and epoch {checkpoint_epoch}')
else:
    checkpoint_step = 0
    checkpoint_epoch = 0

def eval(model, dataloader, criterion, sop=True, binary_threshold=-1):
    print('Eval ...')
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    total_nnz = 0
    total_num_masks = 0
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

                attention_mask = batch['attention_mask'].to(device)
                segs = batch['segs'].to(device).float()
            kwargs = {
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
            }
            labels = batch['label'].to(device)
            # import pdb; pdb.set_trace()

            if sop:
                outputs = model(inputs, segs=segs, kwargs=kwargs, return_tuple=True,
                                binary_threshold=binary_threshold)

                logits = outputs.logits

                for i in range(len(logits)):
                    pred = logits[i].argmax(-1).item()

                    pred_mask_idxs_sort = outputs.mask_weights[i,:,pred].argsort(descending=True)
                    mask_weights_sort = (outputs.mask_weights * outputs.logits_all)[i,pred_mask_idxs_sort,pred]
                    masks_sort = outputs.masks[0,pred_mask_idxs_sort]
                    masks_sort_used = (masks_sort[mask_weights_sort != 0] > masks_sort[mask_weights_sort != 0].mean()).int()
                    mask_weights_sort_used = mask_weights_sort[mask_weights_sort > 0]
                    nnz = (masks_sort[mask_weights_sort != 0] > 0).sum() / masks_sort[mask_weights_sort != 0].view(-1).shape[0]
                    total_nnz += nnz.item()
                    total_num_masks += len(masks_sort_used)
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
    val_nnz = total_nnz / total
    val_n_masks_avg = total_num_masks / total
    
    model.train()
    
    return {
        'val_acc': val_acc,
        'val_loss': val_loss,
        'val_nnz': val_nnz,
        'val_n_masks_avg': val_n_masks_avg
    }

backbone_val_results = eval(wrapped_backbone_model, val_dataloader, criterion, sop=False)
backbone_val_acc = backbone_val_results['val_acc']

import logging
logging.basicConfig(filename=os.path.join(exp_dir, 'train.log'), level=logging.INFO)

log_message = f'Backbone val acc {backbone_val_acc:.4f}'
print(log_message)
logging.info(log_message)

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

model.train()

progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    if epoch < checkpoint_epoch:
        step += len(train_dataloader)
        progress_bar.update(len(train_dataloader))
        continue
    running_loss = 0.0
    running_total = 0
    for i, batch in enumerate(train_dataloader):
        if step <= checkpoint_step:
            step += 1
            progress_bar.update(1)
            continue
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
        train_rep_step = step // train_rep_step_size
        if args.model == 'sop':
            logits = model(inputs, segs=segs, epoch=train_rep_step, mask_batch_size=mask_batch_size, kwargs=kwargs)
        else: # fresh
            logits = model(inputs, **kwargs).logits
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
            if args.model == 'sop':
                val_results = eval(model, val_dataloader, criterion)
                val_results_bin = eval(model, val_dataloader, criterion, binary_threshold=0.5)
                val_acc_bin = val_results_bin['val_acc']
                val_loss_bin = val_results_bin['val_loss']
                val_nnz_bin = val_results_bin['val_nnz']
                val_n_masks_avg_bin = val_results_bin['val_n_masks_avg']
            else:
                val_results = eval(model, val_dataloader, criterion, sop=False)
                val_acc_bin = None
                val_loss_bin = None
                val_nnz_bin = None
                val_n_masks_avg_bin = None
            val_acc = val_results['val_acc']
            val_loss = val_results['val_loss']
            val_nnz = val_results['val_nnz']
            val_n_masks_avg = val_results['val_n_masks_avg']
            log_message = f'Epoch {epoch}, Step {step}, Val acc {val_acc:.4f}, Val loss {val_loss:.4f}'
            log_message += f', Val nnz {val_nnz:.4f}, Val n masks avg {val_n_masks_avg:.4f}'
            print(log_message)
            logging.info(log_message)
            if track:
                wandb.log({'val_acc': val_acc,
                           'val_loss': val_loss,
                           'val_nnz': val_nnz,
                            'val_n_masks_avg': val_n_masks_avg,
                            'val_acc_bin': val_acc_bin,
                            'val_loss_bin': val_loss_bin,
                            'val_nnz_bin': val_nnz_bin,
                            'val_n_masks_avg_bin': val_n_masks_avg_bin,
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
                torch.save(checkpoint, best_checkpoint_path)
                config.save_to_json(config_best_checkpoint_path)
                print(f'Best checkpoint saved at {best_checkpoint_path}')
                
                # model.save_pretrained(best_dir)
            # model.save_pretrained(last_dir)
            os.makedirs(last_dir, exist_ok=True)
            torch.save(checkpoint, last_checkpoint_path)
            config.save_to_json(config_best_checkpoint_path)
            print(f'Last checkpoint saved at {last_checkpoint_path}')
            
        lr_scheduler.step()
        progress_bar.update(1)
        
        step += 1

if args.model == 'sop':
    model.save(exp_dir)
else:
    model.save_pretrained(exp_dir)