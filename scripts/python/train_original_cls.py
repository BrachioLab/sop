import json
import os
import argparse
import wandb

import numpy as np
import random
import torch
from torch import nn, optim
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pathlib import Path
from torch.utils.data import DataLoader
from transformers import get_scheduler

def parse_args():
    parser = argparse.ArgumentParser()

    # models and configs
    parser.add_argument('--blackbox-model-name', type=str, 
                        default='google/vit-base-patch16-224', 
                        help='black box model name')
    parser.add_argument('--blackbox-processor-name', type=str, 
                        default='google/vit-base-patch16-224', 
                        help='black box processor name')
    parser.add_argument('--wrapper-config-filepath', type=str, 
                        default='actions/wrapper/configs/imagenet_vit_wrapper_default.json', 
                        help='wrapper config file')
    parser.add_argument('--exp-dir', type=str, 
                        default='exps/imgenet_wrapper', 
                        help='exp dir')
    parser.add_argument('--wrapper-model', type=str, 
                        default='sop', choices=['sop', 'fresh'],
                        help='sop or fresh model')
    parser.add_argument('--model-type', type=str, 
                        default='image', choices=['image', 'text'],
                        help='image or text model')
    parser.add_argument('--projection-layer-name', type=str, 
                        default=None,  #distilbert.embeddings
                        help='projection layer if specified, else train from scratch')
    
    # data
    parser.add_argument('--dataset', type=str, 
                        default='imagenet', choices=['imagenet', 
                                                     'imagenet_m',
                                                     'multirc',
                                                     'movies'],
                        help='which dataset to use')
    parser.add_argument('--train-size', type=int, 
                        default=-1, 
                        help='-1: use all, else randomly choose k per class')
    parser.add_argument('--val-size', type=int, 
                        default=-1, 
                        help='-1: use all, else randomly choose k per class')
    
    # training
    parser.add_argument('--num-epochs', type=int, 
                        default=1, 
                        help='num epochs')
    parser.add_argument('--num-train-reps', type=int, 
                        default=1, 
                        help='number of times to train each head')
    parser.add_argument('--lr', type=float, 
                        default=1e-4, 
                        help='learning rate')
    parser.add_argument('--lr-scheduler-step-size', type=int, 
                        default=1, 
                        help='learning rate scheduler step size (by epoch)')
    parser.add_argument('--lr-scheduler-gamma', type=float, 
                        default=0.1, 
                        help='learning rate scheduler gamma')
    parser.add_argument('--warmup-epochs', type=float, 
                        default=-1, 
                        help='number of epochs to warmup')
    parser.add_argument('--warmup-steps', type=float, 
                        default=0, 
                        help='number of steps to warmup. Use this when warmup epochs is -1')
    parser.add_argument('--scheduler-type', type=str, 
                        default='inverse_sqrt_heads', choices=['cosine',
                                                               'constant',
                                                               'inverse_sqrt_heads'],
                        help='scheduler type')
    parser.add_argument('--criterion', type=str, 
                        default='ce', choices=['ce', 'bce'],
                        help='which criterion to use, if multi-label then bce')
    parser.add_argument('--batch-size', type=int, 
                        default=32, 
                        help='batch size')
    parser.add_argument('--mask-batch-size', type=int, 
                        default=16, 
                        help='mask batch size')
    parser.add_argument('--project-name', type=str, 
                        default='attn', 
                        help='wandb project name')
    parser.add_argument('--seed', type=int, 
                        default=42, 
                        help='seed')
    parser.add_argument('--track', action='store_true', 
                        default=False, 
                        help='track')
    parser.add_argument('--num-masks-sample-inference', type=int, 
                        default=None, 
                        help='number of masks to retain for mask dropout.')

    # specify wrapper config. If not None, then use this instead of ones specified in config
    parser.add_argument('--attn-patch-size', type=int, 
                        default=None, 
                        help='attn patch size, does not have to match black box model patch size')
    parser.add_argument('--num-heads', type=int, 
                        default=None, 
                        help='hidden dim for the first attention layer')
    parser.add_argument('--num-masks-sample', type=int, 
                        default=None, 
                        help='number of masks to retain for mask dropout.')
    parser.add_argument('--num-masks-max', type=int, 
                        default=None, 
                        help='number of maximum masks to retain.')
    parser.add_argument('--finetune-layers', type=str, 
                        default=None, 
                        help='Which layer to finetune, seperated by comma.')
    
    parser.add_argument('--aggr-type', type=str, 
                        default='joint', choices=['joint', 'independent'],
                        help='usually we use joint for classification, but can also try independent')
    parser.add_argument('--num-masks-inference', type=int, 
                        default=-1, 
                        help='number of masks to retain for mask dropout during inference.' +
                             ' default to -1 which is using all')
    
    return parser

def main():
    parser = parse_args()
    args = parser.parse_args()

    print('\n---argparser---:')
    for arg in vars(args):
        print(arg, getattr(args, arg), '\t', type(arg))

    os.makedirs(args.exp_dir, exist_ok=True)

    # Set the seed for reproducibility
    if args.seed != -1:
        # Torch RNG
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # Python RNG
        np.random.seed(args.seed)
        random.seed(args.seed)

    print(f'Project name {args.project_name}\n')
    if args.track:
        wandb.init(project=args.project_name)
        wandb.run.name = args.exp_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define your dataset and dataloader
    if args.model_type == 'image':
        processor = AutoImageProcessor.from_pretrained(args.blackbox_processor_name)
    else:
        processor = AutoTokenizer.from_pretrained(args.blackbox_processor_name)

    # merge blackbox config and wrapper config
    config = AutoConfig.from_pretrained(args.blackbox_model_name)

    train_dataset, val_dataset = get_datasets(args.dataset, 
                                processor, 
                                debug=False,
                                # transform=transform,
                                train_size=args.train_size,
                                val_size=args.val_size,
                                label2id=config.label2id)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)


    if args.model_type == 'image':
        inner_model = AutoModelForImageClassification.from_pretrained(args.blackbox_model_name)
    else:
        inner_model = AutoModelForSequenceClassification.from_pretrained(args.blackbox_model_name)
    
    model = inner_model

    if args.criterion == 'ce':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=args.lr)

    num_training_steps = args.num_epochs * len(train_loader)

    if args.warmup_epochs != -1:
        num_warmup_steps=int(len(train_loader) * args.warmup_epochs)
    else:
        num_warmup_steps=args.warmup_steps

    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )


    # Define your training loop
    best_val_acc = 0.0
    
    progress_bar = tqdm(range(num_training_steps))
    step = 0
    for epoch in range(args.num_epochs):
        running_loss = 0.0
        model.train()
        
        for i, batch in enumerate(train_loader):
            # import pdb
            # pdb.set_trace()
            
            if args.model_type == 'image':
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                token_type_ids = None
                attention_mask = None
            else:
                if not isinstance(batch['input_ids'], torch.Tensor):
                    inputs = torch.stack(batch['input_ids']).transpose(0, 1).to(device)
                    token_type_ids = torch.stack(batch['token_type_ids']).transpose(0, 1).to(device)
                    attention_mask = torch.stack(batch['attention_mask']).transpose(0, 1).to(device)
                else:
                    inputs = batch['input_ids'].to(device)
                    token_type_ids = batch['token_type_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
            optimizer.zero_grad()
            
            if args.model_type == 'image':
                outputs = model(inputs)
            else: # text
                outputs = model(inputs, 
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask)

            logits = outputs.logits
            
            loss = criterion(logits, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            progress_bar.update(1)

            running_loss += loss.item()

            if i % 100 == 99 or i == len(train_loader) - 1:
                # Print training loss every 100 batches
                curr_lr = float(optimizer.param_groups[0]['lr'])
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss {running_loss / 100:.4f}, LR {curr_lr:.5f}')
                if args.track:
                    wandb.log({'train_loss': running_loss / 100,
                            'lr': curr_lr,
                            'epoch': epoch,
                            'step': step})
                running_loss = 0.0

            if i % 1000 == 999 or i == len(train_loader) - 1:
                correct = 0
                total = 0
                print('Eval..')
                model.eval()
                with torch.no_grad():
                    progress_bar_eval = tqdm(range(len(val_loader)))
                    for batch in val_loader:
                        if args.model_type == 'image':
                            inputs, labels = batch
                            inputs, labels = inputs.to(device), labels.to(device)
                            token_type_ids = None
                            attention_mask = None
                        else:
                            if not isinstance(batch['input_ids'], torch.Tensor):
                                inputs = torch.stack(batch['input_ids']).transpose(0, 1).to(device)
                                token_type_ids = torch.stack(batch['token_type_ids']).transpose(0, 1).to(device)
                                attention_mask = torch.stack(batch['attention_mask']).transpose(0, 1).to(device)
                            else:
                                inputs = batch['input_ids'].to(device)
                                token_type_ids = batch['token_type_ids'].to(device)
                                attention_mask = batch['attention_mask'].to(device)
                            labels = batch['label'].to(device)
                        
                        if args.model_type == 'image':
                            outputs = model(inputs)
                        else: # text
                            outputs = model(inputs, 
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)

                        logits = outputs.logits

                        if args.criterion == 'ce':
                            _, predicted = torch.max(logits.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                        else:
                            probs = torch.sigmoid(logits)
                            predicted = (probs > 0.5).float()
                            correct += torch.sum((predicted == labels).all(dim=1)).item()  # all pred correct
                            total += labels.size(0)
                        
                        progress_bar_eval.update(1)
                        
                val_acc = 100 * correct / total
                print(f'Epoch {epoch + 1}, Step {step}, Validation Accuracy {val_acc:.2f}%')

                last_dir = os.path.join(args.exp_dir, 'last')
                best_dir = os.path.join(args.exp_dir, 'best')
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    model.save_pretrained(best_dir)
                model.save_pretrained(last_dir)
                if args.track:
                    wandb.log({'val_acc': val_acc,
                            'epoch': epoch,
                            'step': step})
                model.train()
                
            lr_scheduler.step()
            step += 1


if __name__ == '__main__':
    main()