import json
import os
import argparse

import numpy as np
import random
import math
import torch
from torch import nn, optim
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pathlib import Path
from torch.utils.data import DataLoader, Subset, Dataset
import sys
sys.path.append('lib/exlib/src')

# import get_scheduler
from transformers import get_scheduler

from PIL import Image
from exlib.modules.sop import SOPImageCls, SOPConfig, get_chained_attr, get_inverse_sqrt_with_separate_heads_schedule_with_warmup
from exlib.modules.fresh import FRESH
from torchvision import transforms


class ImageFolderSubDataset(Dataset):
    def __init__(self, data_dir, transform=None, num_data=-1):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for label in sorted(os.listdir(data_dir)):
            dirname = os.path.join(data_dir, label)
            if not os.path.isdir(dirname):
                continue
            for i, image_path in enumerate(sorted(os.listdir(dirname))):
                if num_data != -1 and i >= num_data:
                    break
                self.image_paths.append(os.path.join(data_dir, label, image_path))
                self.labels.append(label)
        self.all_labels = sorted(list(set(self.labels)))
        
        print('Loaded {} images and {} classes'.format(len(self.image_paths), len(self.all_labels)))

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.all_labels.index(self.labels[idx])
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def parse_args():
    parser = argparse.ArgumentParser()

    # paths and info
    parser.add_argument('--lr', type=float, 
                        default=0.000005,
                        help='lr')
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
                        default=2,
                        help='num heads')
    parser.add_argument('--batch-size', type=int, 
                        default=16,
                        help='num heads')
    parser.add_argument('--train-size', type=int,
                        default=-1,
                        help='num data for train')
    parser.add_argument('--val-size', type=int,
                        default=-1,
                        help='num data for val')
    parser.add_argument('--scheduler-type', type=str,
                        default='inv_sqrt', choices=['inv_sqrt', 'cosine'],
                        help='scheduler type')
    parser.add_argument('--num-epochs', type=float,
                        default=1,
                        help='num epochs')
    parser.add_argument('--model', type=str, 
                        default='sop', choices=['sop', 'fresh'],
                        help='use sop or fresh model')
    parser.add_argument('--backbone-model', type=str, 
                        default='vit', choices=['vit', 'resnet'],
                        help='backbone model')
    parser.add_argument('--dataset', type=str, 
                        default='imagenet', choices=['imagenet', 'imagenet_m'],
                        help='num heads')
    parser.add_argument('--reload-checkpoint', 
                        default=False, action='store_true',
                        help='if true, then reload checkpoint')
    parser.add_argument('--ft-backbone', 
                        default=False,
                        action='store_true',
                        help='finetune backbone')
    
    
    return parser

parser = parse_args()
args = parser.parse_args()

LR = args.lr
GROUP_GEN_SCALE = args.group_gen_scale
GROUP_SEL_SCALE = args.group_sel_scale
GROUP_GEN_TEMP_ALPHA = args.group_gen_temp_alpha
GROUP_GEN_TEMP_BETA = args.group_gen_temp_beta
NUM_HEADS = args.num_heads
TRAIN_SIZE = args.train_size
VAL_SIZE = args.val_size
NUM_EPOCHS = args.num_epochs
ft_backbone = '_ft' if args.ft_backbone else ''

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
if args.dataset == 'imagenet':
    backbone_model_name = 'google/vit-base-patch16-224'
else:
    backbone_model_name = 'pt_models/vit-base-patch16-224-imagenet10cls'
backbone_processor_name = 'google/vit-base-patch16-224'
# sop_config_path = 'configs/imagenet_m.json'

# data paths
if args.dataset == 'imagenet':
    TRAIN_DATA_DIR = 'data/imagenet/train'
    VAL_DATA_DIR = 'data/imagenet/val'
else:
    TRAIN_DATA_DIR = 'data/imagenet_m/train'
    VAL_DATA_DIR = 'data/imagenet_m/val'

# training args
batch_size = args.batch_size
lr = LR
num_epochs = NUM_EPOCHS
warmup_steps = 2000
mask_batch_size = 64
group_gen_scale = GROUP_GEN_SCALE
group_sel_scale = GROUP_SEL_SCALE
# group_gen_temperature = GROUP_GEN_TEMPERATURE
group_gen_temp_alpha = GROUP_GEN_TEMP_ALPHA
group_gen_temp_beta = GROUP_GEN_TEMP_BETA
# backbone_layer = BACKBONE_LAYER
train_size = TRAIN_SIZE
val_size = VAL_SIZE
# freeze_projection = FREEZE_PROJECTION

# experiment args
if args.dataset == 'imagenet':
    exp_name = 'imagenet_bbm{}_{}h_lr{}_gg{}_gs{}_ggta{}_ggtb{}_ts{}_vs{}_st{}_ep{}{}'.format(args.backbone_model,
                NUM_HEADS, 
                lr, group_gen_scale, group_sel_scale, group_gen_temp_alpha, group_gen_temp_beta,
                train_size, val_size, args.scheduler_type, num_epochs, ft_backbone)
else:
    exp_name = 'imagenet_m_bbm{}_{}h_lr{}_gg{}_gs{}_ggta{}_ggtb{}_ts{}_vs{}_st{}_ep{}{}'.format(args.backbone_model,
                NUM_HEADS, 
                lr, group_gen_scale, group_sel_scale, group_gen_temp_alpha, group_gen_temp_beta,
                train_size, val_size, args.scheduler_type, num_epochs, ft_backbone)

if args.model == 'fresh':
    exp_name += '_fresh'

exp_dir = os.path.join('exps', exp_name)

os.makedirs(exp_dir, exist_ok=True)

# checkpoint dirs
last_dir = os.path.join(exp_dir, 'last')
best_dir = os.path.join(exp_dir, 'best')
best_checkpoint_path = os.path.join(best_dir, 'checkpoint.pth')
last_checkpoint_path = os.path.join(last_dir, 'checkpoint.pth')
config_best_checkpoint_path = os.path.join(best_dir, 'config.json')
config_last_checkpoint_path = os.path.join(last_dir, 'config.json')

if args.backbone_model == 'vit':
    backbone_model = AutoModelForImageClassification.from_pretrained(backbone_model_name)
    processor = AutoImageProcessor.from_pretrained(backbone_processor_name)
    backbone_config = AutoConfig.from_pretrained(backbone_model_name)
else: # resnet
    backbone_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).to(device)
    processor = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


config = SOPConfig(
    attn_patch_size=16,
    num_heads=NUM_HEADS,
    num_masks_sample=20,
    num_masks_max=200,
    finetune_layers=['model.classifier'] if args.backbone_model == 'vit' else ['model.fc'],
    group_gen_scale=group_gen_scale,
    group_sel_scale=group_sel_scale,
    group_gen_temp_alpha=group_gen_temp_alpha,
    group_gen_temp_beta=group_gen_temp_beta,
)
print('config', config.__dict__)
if args.backbone_model == 'vit':
    config.__dict__.update(backbone_config.__dict__)
    config.num_labels = len(backbone_config.id2label)
else:
    if args.dataset == 'imagenet':
        config.num_labels = 1000
    else:
        config.num_labels = 10
# config.save_pretrained(exp_dir)

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

if args.backbone_model == 'vit':
    def transform(image):
        # Preprocess the image using the ViTImageProcessor
        image = image.convert("RGB")
        inputs = processor(image, return_tensors='pt')
        return inputs['pixel_values'].squeeze(0)
else:
    def transform(image):
        # Preprocess the image using the ResNet
        image = image.convert("RGB")
        inputs = processor(image)
        return inputs

# Load the dataset
train_dataset = ImageFolderSubDataset(TRAIN_DATA_DIR, transform=transform, num_data=train_size)
val_dataset = ImageFolderSubDataset(VAL_DATA_DIR, transform=transform, num_data=val_size)

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

if args.backbone_model == 'vit':
    class WrappedBackboneModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, inputs):
            outputs = self.model(inputs, output_hidden_states=True)
            return WrappedBackboneOutput(outputs.logits, outputs.hidden_states[-1][:,0])
else:
    class WrappedBackboneModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def _forward_impl(self, x):
            # See note [TorchScript super()]
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            x = self.model.avgpool(x)
            hidden_states = torch.flatten(x, 1)
            x = self.model.fc(hidden_states)

            return x, hidden_states
        
        def forward(self, inputs):
            # outputs = self.model(inputs, output_hidden_states=True)
            outputs = self._forward_impl(inputs)
            return WrappedBackboneOutput(outputs[0], outputs[1])


wrapped_backbone_model = WrappedBackboneModel(backbone_model)
wrapped_backbone_model = wrapped_backbone_model.to(device)
# class_weights = get_chained_attr(wrapped_backbone_model, config.finetune_layers[0]).weight #.clone().to(device)

if args.model == 'sop':
    model = SOPImageCls(config, wrapped_backbone_model)
else:
    import copy
    fresh_config = copy.deepcopy(backbone_config)
    fresh_config.__dict__.update(config.__dict__)
    fresh_config.finetune_layers = [fresh_config.finetune_layers[0].replace('model.', '')]
    model = FRESH(fresh_config,
                    backbone_model,
                    model_type='image',
                    return_tuple=True,
                    postprocess_attn=lambda x: x.attentions[-1].mean(dim=1)[:,0],
                    postprocess_logits=lambda x: x.logits)
model = model.to(device)

if args.ft_backbone:
    for name, param in model.blackbox_model.named_parameters():
        param.requires_grad = True

from transformers import get_scheduler

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
num_training_steps = int(len(train_dataloader) * num_epochs)
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
                "cosine",
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
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            if sop:
                outputs = model(inputs, return_tuple=True, binary_threshold=binary_threshold)
                
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
                outputs = model(inputs)
                logits = outputs.logits
            
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
print('Backbone val acc: ', backbone_val_acc)


import logging

track = True
# track = False

if track:
    import wandb
    wandb.init(project='sop')
    wandb.run.name = os.path.basename(exp_dir)
    wandb.log({'backbone_val_acc': backbone_val_acc})

# Iterate over the data
best_val_acc = 0.0
step = 0
train_log_interval = 100
val_eval_interval = 1000

logging.basicConfig(filename=os.path.join(exp_dir, 'train.log'), level=logging.INFO)


progress_bar = tqdm(range(num_training_steps))
for epoch in range(math.ceil(num_epochs)):
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
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        train_rep_step = step // train_rep_step_size
        if args.model == 'sop':
            logits = model(inputs, epoch=train_rep_step, mask_batch_size=mask_batch_size)
        else:
            logits = model(inputs).logits
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
            
        # if i % val_eval_interval == val_eval_interval - 1 or i == len(train_dataloader) - 1:
        if i % val_eval_interval == min(100, val_eval_interval - 1) or i == len(train_dataloader) - 1:
            if args.model == 'sop':
                val_results = eval(model, val_dataloader, criterion)
                val_results_bin = eval(model, val_dataloader, criterion, binary_threshold=0.5)
            else:
                val_results = eval(model, val_dataloader, criterion, sop=False)
            val_acc = val_results['val_acc']
            val_loss = val_results['val_loss']
            val_nnz = val_results['val_nnz']
            val_n_masks_avg = val_results['val_n_masks_avg']
            val_acc_bin = val_results_bin['val_acc']
            val_loss_bin = val_results_bin['val_loss']
            val_nnz_bin = val_results_bin['val_nnz']
            val_n_masks_avg_bin = val_results_bin['val_n_masks_avg']
            log_message = f'Epoch {epoch}, Step {step}, Val acc {val_acc:.4f}, Val loss {val_loss:.4f}'
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
        if step > num_training_steps:
            break
    if step > num_training_steps:
        break    
    
if args.model == 'sop':
    model.save(exp_dir)
else:
    model.save_pretrained(exp_dir)