from collections import defaultdict
from tqdm.auto import tqdm
from ..utils.data_utils import ImageFolderSegSubDataset
from exlib.evaluators.common import convert_idx_masks_to_bool
import numpy as np
from ..utils import get_explainer, get_dataset
import torch
import math


def get_k_pred(explainer, method, inputs, attrs, ins_steps, start, end, original_preds, 
               use_original_pred=False, deletion=False, return_all=False):
    device = inputs.device
    method_list = method.split('_')
    explainer_name = method_list[0]
    
    # get masks
    masks_all = []
    for idx in range(len(inputs)):

        # Create a mask of size (14, 14) with values from 1 to 14*14
        if method == 'bagnet':
            print('recompute')
            expln = explainer(inputs, return_groups=True)
            masks = expln.group_masks[0]
            mask_weights = expln.group_attributions[0].flatten()
        else:
            cell_size = 14
            image_size = 224
            mask = torch.arange(1, cell_size*cell_size + 1, dtype=torch.int).reshape(cell_size, cell_size)

            # Resize the mask to (224, 224) without using intermediate floating point numbers
            # This can be achieved by repeating each value in both dimensions to scale up the mask
            scale_factor = image_size // cell_size  # Calculate scale factor
            resized_mask = mask.repeat_interleave(scale_factor, dim=0).repeat_interleave(scale_factor, dim=1)

            masks = convert_idx_masks_to_bool(resized_mask[None]).to(device)
            mask_weights = (masks.to(device) * attrs[idx][0:1].to(device)).sum(-1).sum(-1).to(device)

        # Sort the masks based on mask_weights
        sort_idxs = torch.argsort(mask_weights, descending=(not deletion))
        masks = masks[sort_idxs]  # Sort masks accordingly
        mask_weights = mask_weights[sort_idxs]

        # Cumulative sum of sorted masks
        cumulative_masks = torch.cumsum(masks, dim=0).bool().float()
        if type(start) is int:
            topks = torch.tensor(np.linspace(start, end, ins_steps), dtype=torch.int).to(masks.device) - 1
        else:
            topks = (torch.tensor(np.linspace(start, end, ins_steps), 
                                  dtype=torch.float).to(masks.device) * 196).int() 
            topks = topks - 1
        masks_use = cumulative_masks[topks]

        masks_all.append(masks_use)

    masks_all = torch.stack(masks_all, dim=0)
    
    # pass into the model and compute insertion scores
    mini_bsz = 32
    masks_all_shape = masks_all.shape
    masks_all_reshape = masks_all.view(-1, masks_all.shape[-2], masks_all.shape[-1])
    
    inputs_repeat = inputs.reshape(inputs.shape[0], 1, 3, inputs.shape[-2], inputs.shape[-1]) \
                    .expand(inputs.shape[0], masks_all_shape[1], 3, inputs.shape[-2], inputs.shape[-1])
    inputs_repeat_reshape = inputs_repeat.reshape(-1, 3, inputs.shape[-2], inputs.shape[-1])
    
    logits = []
    for mini_bi in range(math.ceil(masks_all_reshape.shape[0] / mini_bsz)):
        # Get masked output
        inputs_mini = inputs_repeat_reshape[mini_bi * mini_bsz:(mini_bi + 1)*mini_bsz]
        masks_mini = masks_all_reshape[mini_bi * mini_bsz:(mini_bi + 1)*mini_bsz]
        masked_inputs = masks_mini[:,None] * inputs_mini
        if explainer_name in ['xdnn', 'bagnet']:
            inputs_norm = explainer.normalize(inputs_mini)
            masked_inputs = masks_mini[:,None] * inputs_norm
        elif explainer_name in ['bcos']:
            inputs_norm = explainer.preprocess(inputs_mini)
            masked_inputs = masks_mini[:,None] * inputs_norm
        
        outputs = explainer.model(masked_inputs)
        
        try:
            logits_mini = outputs.logits
        except:
            logits_mini = outputs
        logits.append(logits_mini)
    logits = torch.cat(logits, dim=0)
    logits = logits.reshape(masks_all_shape[0], masks_all_shape[1], logits.shape[-1])

    preds = logits.argmax(-1)
    probs = logits.softmax(-1)

    # if use_original_pred:
    return probs[range(len(preds)), :, original_preds]

def insertion(model, original_model, backbone_model, processor, val_config,
              method, ins_steps, start, end, 
              # use_original_pred=False, 
              debug=False, 
              deletion=False, return_all=False):
    device = backbone_model.device
    print(method)
    method_list = method.split('_')
    explainer_name = method_list[0]
    # if explainer_name == 'backbone':
    #     explainer_name = 'attn'
    if len(method_list) == 2:
        suffix = f'_{method_list[1]}'
    else:
        suffix = ''
    
    if explainer_name in ['bagnet', 'sop']:
        ATTR_VAL_DATA_DIR = None
    else:
        ATTR_VAL_DATA_DIR = f'/shared_data0/weiqiuy/sop/exps/imagenet_vit_1/attributions_seg/{explainer_name.replace("-s", "")}_1_pred{suffix}/val'

    if method == 'sop':
        explainer = model
    else:
        explainer = get_explainer(original_model, backbone_model, method.split('_')[0], device)
    
    val_dataset, val_dataloader = get_dataset(val_config['dataset']['name'], 
                                          split=val_config['evaluation']['split'], 
                                          num_data=val_config['evaluation']['num_data'],
                                          batch_size=val_config['evaluation']['batch_size'],
                                                        attr_dir=ATTR_VAL_DATA_DIR,
                                          processor=processor, debug=debug)

    insertion_scores = []
    insertion_scores_curve = []
    for bi, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        if debug and bi >= int(debug):
            break
        if len(batch) == 5:
            inputs, labels, segs, attrs, idxs = batch
        else:
            inputs, labels, segs, idxs = batch
            attrs = None
        inputs, labels, segs = inputs.to(device), labels.to(device), segs.to(device)
        with torch.no_grad():
            if explainer_name == 'sop':
                model.k = 0.2
                logits = model(inputs)
                original_preds = logits.argmax(-1)
                
                ins_scores_bi = []
                for k in tqdm(np.linspace(start, end, ins_steps)):
                    # k = int(k * 196)
                    if type(start) is int and type(end) is int and type(ins_steps) is int:
                        k = int(k)
                    model.k = k
                    # wrapped_backbone_model.model(inputs, output_hidden_states=True)
                    outputs = model(inputs, return_tuple=True, binary_threshold=-1, deletion=deletion)
                    logits = outputs.logits
                    preds = logits.argmax(-1)
                    probs = logits.softmax(-1)
                    ins_scores_bi.append(probs[range(len(original_preds)), preds]) #original_preds])
                ins_scores_bi = torch.stack(ins_scores_bi, 0).transpose(0, 1)
                # print('ins_scores_bi', ins_scores_bi.shape)
                results = ins_scores_bi
                # print(results)
            else:
                if explainer_name in ['xdnn', 'bagnet']:
                    inputs_norm = explainer.normalize(inputs)
                    # original_logits = explainer.model(inputs_norm)
                elif explainer_name in ['bcos']:
                    inputs_norm = explainer.preprocess(inputs)
                # elif explainer_name in ['attn']: todo: for attn, pred should be new pred
                else:
                    inputs_norm = inputs

                original_logits = explainer.model(inputs_norm)
                if 'logits' in original_logits.__dict__:
                    original_logits = original_logits.logits
                original_preds = torch.argmax(original_logits, dim=-1)

                results = get_k_pred(explainer, method, inputs, attrs, ins_steps, start, end, original_preds, 
                                     deletion=deletion, return_all=return_all)
            insertion_scores.extend(results.mean(1).tolist())
            insertion_scores_curve.extend(results.tolist())
        
    insertion_scores_perc = []
    for i in range(len(insertion_scores_curve)):
        mean_ins = np.mean(insertion_scores_curve[i])
        max_ins = insertion_scores_curve[i][-1] # np.max(ins_scores_all_curve[i])
        insertion_scores_perc.append(mean_ins / max_ins )
    return insertion_scores, insertion_scores_curve, insertion_scores_perc

def get_ins_del_perc(val_config, original_model, backbone_model, model, processor,
                     method, ins_steps=10, start=0.1, end=1.0, deletion=False, debug=False, vis=False):
    ins_scores, ins_scores_curve, insertion_scores_perc = insertion(model, 
                                                                          original_model, 
                                                                          backbone_model, 
                                                                          processor, 
                                                                          val_config,
                                                                          method, 
                                             ins_steps=ins_steps, start=start, end=end, 
                                             debug=debug, return_all=True, deletion=deletion)
    if vis:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(ins_scores_original_curve[0])
        plt.show()
    print(np.mean(insertion_scores_perc))
    return {
        'scores_mean': ins_scores,
        'scores_curve': ins_scores_curve,
        'scores_perc': insertion_scores_perc,
    }