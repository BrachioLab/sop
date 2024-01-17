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
from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image


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
    backbone_model_name = 'pt_models/multirc_vanilla/best'
    backbone_processor_name = 'bert-base-uncased'
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
    exp_dir = 'exps/multirc_5e-06/best'

    backbone_model = AutoModelForSequenceClassification.from_pretrained(backbone_model_name)
    processor = AutoTokenizer.from_pretrained(backbone_processor_name)
    backbone_config = AutoConfig.from_pretrained(backbone_model_name)

    config = SOPConfig(json_file=os.path.join(exp_dir, 'config.json'),
                    projected_input_scale=2)

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

    train_size, val_size = 100, 100

    # train_dataset = load_dataset('eraser_multi_rc', split='train')
    # train_dataset = train_dataset.map(transform, batched=True,
    #                             remove_columns=['passage', 
    #                                             'query_and_answer',
    #                                             'evidences'])

    val_dataset = load_dataset('eraser_multi_rc', split='validation')
    val_dataset = val_dataset.map(transform, batched=True,
                                remove_columns=['passage', 
                                                'query_and_answer',
                                                'evidences'])

    # if train_size != -1:
    #     train_dataset = Subset(train_dataset, list(range(train_size)))
    if num_data != -1:
        val_dataset = Subset(val_dataset, list(range(num_data)))

    # Create a DataLoader to batch and shuffle the data
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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
        inputs_dict = {
            'input_ids': inputs,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        }
        labels = batch['label'].to(device)
        bsz = inputs.shape[0]
        with torch.no_grad():
            # logits = original_model(**inputs_dict)
            expln = model(inputs, segs=segs, kwargs=kwargs, return_tuple=True)
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
