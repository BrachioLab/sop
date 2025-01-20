import torch
import os
from transformers import AutoImageProcessor, AutoModelForImageClassification
import argparse

import sys
sys.path.append('/shared_data0/weiqiuy/exlib/src')
sys.path.append('/shared_data0/weiqiuy/sop/src')
from exlib.modules.sop import WrappedModel, SOPConfig, SOPImageCls, get_chained_attr

from exlib.modules.sop import get_masks_used
from tqdm.auto import tqdm
from sop.tasks.images.imagenet import get_explainer



from sop.tasks.images.imagenet.data import get_dataset

from exlib.modules.sop import convert_idx_masks_to_bool
import math

def get_mask(explainer, inputs, expln, method, k=0.2, cell_size=14, image_size=224, idx=0, deletion=False, device='cuda'):
    # Create a mask of size (14, 14) with values from 1 to 14*14
    if method == 'bagnet':
        # print('recompute')
        # expln = explainer(inputs[idx:idx+1], return_groups=True)
        masks = expln.group_masks[idx]
        mask_weights = expln.group_attributions[idx].flatten()
    else:
        # cell_size = 14
        # image_size = 224
        attrs = expln.attributions
        mask = torch.arange(1, cell_size*cell_size + 1, dtype=torch.int).reshape(cell_size, cell_size)

        # Resize the mask to (224, 224) without using intermediate floating point numbers
        # This can be achieved by repeating each value in both dimensions to scale up the mask
        scale_factor = image_size // cell_size  # Calculate scale factor
        resized_mask = mask.repeat_interleave(scale_factor, dim=0).repeat_interleave(scale_factor, dim=1)

        masks = convert_idx_masks_to_bool(resized_mask[None]).to(device)
        # print('attrs', attrs.shape)
        mask_weights = (masks.to(device) * attrs[idx][0:1].to(device)).sum(-1).sum(-1).to(device)

    # Sort the masks based on mask_weights
    sort_idxs = torch.argsort(mask_weights, descending=(not deletion))
    masks = masks[sort_idxs]  # Sort masks accordingly
    mask_weights = mask_weights[sort_idxs]

    # Cumulative sum of sorted masks
    masks_cumsum = torch.cumsum(masks, dim=0).bool().float()

    # print('masks', masks.shape)
    if method == 'bagnet':
        # total_masks = masks_cumsum.shape[0]
        total_pixels = masks_cumsum.shape[-1] * masks_cumsum.shape[-2]

        # start_new = start / end
        # end_new = 1.0

        # if type(start) is int:
        #     topks = torch.tensor(np.linspace(start, end, ins_steps), dtype=torch.int).to(masks.device) - 1
        # else:
        # topks = (torch.tensor(np.linspace(start_new, end_new, ins_steps), 
        #                     dtype=torch.float).to(masks.device) * total_pixels).int() 
        # topks = topks - 1
        topks = math.ceil(k * total_pixels)
        masks_cumsum_sum = masks_cumsum.sum((-1, -2))
        # print('topks', topks)
        # print('masks_cumsum_sum', masks_cumsum_sum)
        # mask_indexs = []
        # for topk in topks:
        mask_indexs = torch.searchsorted(masks_cumsum_sum, topks)
        # import pdb; pdb.set_trace()
        mask_indexs = torch.clamp(mask_indexs, max=masks_cumsum.shape[0] - 1)

        topks = mask_indexs
    else:
        topks = math.ceil(k * len(masks))
        # if type(start) is int:
        #     topks = torch.tensor(np.linspace(start, end, ins_steps), dtype=torch.int).to(masks.device) - 1
        # else:
        #     topks = (torch.tensor(np.linspace(start, end, ins_steps), 
        #                         dtype=torch.float).to(masks.device) * 196).int() 
        #     topks = topks - 1

    masks_use = masks_cumsum[topks]
    return masks_use


# get largest masks for the batch
def get_top_mask_batch(outputs):
    preds = outputs.logits.argmax(-1)
    batch_indices = torch.arange(len(preds))
    pred_mask_idxs_sort = outputs.mask_weights[batch_indices,:,preds].argsort(descending=True)
    top_mask_indices = pred_mask_idxs_sort[:, 0]
    largest_masks = outputs.masks[batch_indices, top_mask_indices]
    return largest_masks

def next_mod(i, div=1000, mod=990):
    return i + (mod - (i % div) + div) % div

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='sop')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--div', type=int, default=1000)
    parser.add_argument('--mod', type=int, default=990)
    parser.add_argument('--skip_saved', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()

    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # init backbone model
    backbone_model = AutoModelForImageClassification.from_pretrained('google/vit-base-patch16-224')
    processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')

    # get needed wrapped models
    original_model = WrappedModel(backbone_model, output_type='logits')
    wrapped_backbone_model = WrappedModel(backbone_model, output_type='tuple')
    projection_layer = WrappedModel(wrapped_backbone_model, output_type='hidden_states')

    # load trained sop model
    model = SOPImageCls.from_pretrained('BrachioLab/sop-vit-base-patch16-224', 
                                        blackbox_model=wrapped_backbone_model, 
                                        projection_layer=projection_layer)

    model = model.to(device)

    model.eval();


    train_dataset, train_dataloader = get_dataset('imagenet', split='train', shuffle=False, processor=processor)
    val_dataset, val_dataloader = get_dataset('imagenet', split='val', num_data=1, shuffle=False, processor=processor)
    train_val_dataset, train_val_dataloader = get_dataset('imagenet', split='train_val', num_data=1, shuffle=False, processor=processor)

    methods = [
        'bcos',
        'xdnn',
        'bagnet',
        'sop',
        'shap',
        'rise',
        'lime',
        'fullgrad',
        'gradcam',
        'intgrad',
        'attn',
        'archipelago',
        'mfaba',
        'agi',
        'ampe',
    ]


    # Example usage
    # i = 1234
    # next_number = next_mod_1000(i)
    # print(next_number)
    # debug = True
    # div = 20
    # mod = 10
    # skip_saved = True

    # print('method', method)
    # method = 'bagnet'
    # for method in methods:

    method = args.method
    debug = args.debug
    div = args.div
    mod = args.mod
    skip_saved = args.skip_saved

    print('method', method)
    print('debug', debug)

    if debug:
        save_dir = f'/shared_data0/weiqiuy/sop/results/groups/imagenet_expln_d{div}_m{mod}_debug/{method}'
        save_dir_attr = f'/shared_data0/weiqiuy/sop/results/groups/imagenet_expln_d{div}_m{mod}_debug_attr/{method}'
    else:
        save_dir = f'/shared_data0/weiqiuy/sop/results/groups/imagenet_expln_d{div}_m{mod}/{method}'
        save_dir_attr = f'/shared_data0/weiqiuy/sop/results/groups/imagenet_expln_d{div}_m{mod}_attr/{method}'

    if method == 'sop':
        explainer = model
    else:
        explainer = get_explainer(original_model, backbone_model, method.split('_')[0], device)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_attr, exist_ok=True)

    max_mask_all = []
    attrs_all = []
    labels_all = []
    for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        # if i % 1000 == 990:
        if skip_saved:
            next_num = next_mod(i, div=div, mod=mod)
            # print('next_num', next_num)
            # print("os.path.exists(os.path.join(save_dir, f'{next_num}.pt'))", os.path.exists(os.path.join(save_dir, f'{next_num}.pt')))
            if os.path.exists(os.path.join(save_dir, f'{next_num}.pt')):
                continue
        if i % div == mod:
            max_mask_all = torch.cat(max_mask_all, dim=0).bool().cpu()
            labels_all = torch.cat(labels_all, dim=0)
            # attrs_all = torch.cat(attrs_all, dim=0)
            torch.save({'max_mask_all': max_mask_all, 'labels_all': labels_all}, 
                        os.path.join(save_dir, f'{i}.pt'))
            torch.save({'attrs_all': attrs_all, 'labels_all': labels_all},
                        os.path.join(save_dir_attr, f'{i}.pt'))
            print('Saved', i)
            max_mask_all = []
            labels_all = []
            attrs_all = []

        if i % 10 != 0:
            continue

        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        if method == 'sop':
            with torch.no_grad():
                outputs = model(inputs.to(device), return_tuple=True)
            masks = get_top_mask_batch(outputs)
            attrs_all.append([outputs.masks, outputs.mask_weights])
        else:
            # outputs = backbone_model(inputs)
            if method in ['xdnn', 'bagnet']:
                inputs_norm = explainer.normalize(inputs)
                # masked_inputs = masks_mini[:,None] * inputs_norm
            elif method in ['bcos']:
                inputs_norm = explainer.preprocess(inputs)
                # masked_inputs = masks_mini[:,None] * inputs_norm
            else:
                inputs_norm = inputs
            if method == 'bagnet':
                with torch.no_grad():
                    outputs = explainer.model(inputs_norm)
            else:
                outputs = explainer.model(inputs_norm)
            if not isinstance(outputs, torch.Tensor):
                logits = outputs.logits
            else:
                logits = outputs
            preds = logits.argmax(-1)
            expln = explainer(inputs, preds)
            masks = []
            for idx in range(inputs.shape[0]):
                mask = get_mask(explainer, inputs, expln, method, idx=idx, device=device)
                masks.append(mask.detach().cpu())
            masks = torch.stack(masks)
            if 'group_attributions' not in expln._fields:
                attrs_all.append(expln.attributions.detach().cpu())
            else:
                attrs_all.append([expln.group_masks.detach().cpu(), expln.group_attributions.detach().cpu()])
        # print(masks.shape)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow((inputs[0].permute(1,2,0).cpu() + 1) / 2)
        # plt.imshow(masks[0].cpu(), alpha=0.5)
        # plt.show()

        max_mask_all.append(masks) # this is the group of the pred, but not necessarily for the label
        labels_all.append(labels)
        if debug and i > 30:
            break
            
    if len(max_mask_all) > 0:
        max_mask_all = torch.cat(max_mask_all, dim=0).bool().cpu()
        labels_all = torch.cat(labels_all, dim=0)
        torch.save({'max_mask_all': max_mask_all, 'labels_all': labels_all}, 
                    os.path.join(save_dir, f'{i}.pt'))
        torch.save({'attrs_all': attrs_all, 'labels_all': labels_all},
                    os.path.join(save_dir_attr, f'{i}.pt'))
        max_mask_all = []
        labels_all = []

    print('Done', method)

if __name__ == '__main__':
    main()