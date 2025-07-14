from collections import defaultdict
from tqdm.auto import tqdm
from ..utils.data_utils import ImageFolderSegSubDataset
from exlib.evaluators.common import convert_idx_masks_to_bool
import numpy as np
# from ..tasks.texts.multirc import get_explainer, get_dataset
from ..tasks.texts import multirc as multirc
from ..tasks.texts import movies as movies

import torch
import math
import os
from exlib.modules.sop_text import gaussian_blur_1d_mask



def get_k_pred_text(explainer, method, inputs, kwargs, attrs, ins_steps, start, end, original_preds, 
               use_original_pred=False, deletion=False, return_all=False, kernel_size=1, sigma=1, debug=False):
    device = inputs.device
    method_list = method.split('_')
    explainer_name = method_list[0]

    attention_mask = kwargs['attention_mask']
    token_type_ids = kwargs['token_type_ids']
    
    
    # get masks
    # masks_all = []
    # for idx in range(len(inputs)):

    #     # Create a mask of size (14, 14) with values from 1 to 14*14
        
    #     cell_size = 14
    #     image_size = 224
    #     mask = torch.arange(1, cell_size*cell_size + 1, dtype=torch.int).reshape(cell_size, cell_size)

    #     # Resize the mask to (224, 224) without using intermediate floating point numbers
    #     # This can be achieved by repeating each value in both dimensions to scale up the mask
    #     scale_factor = image_size // cell_size  # Calculate scale factor
    #     resized_mask = mask.repeat_interleave(scale_factor, dim=0).repeat_interleave(scale_factor, dim=1)

    #     masks = convert_idx_masks_to_bool(resized_mask[None]).to(device)
    #     mask_weights = (masks.to(device) * attrs[idx][0:1].to(device)).sum(-1).sum(-1).to(device)

    #     # Sort the masks based on mask_weights
    #     sort_idxs = torch.argsort(mask_weights, descending=(not deletion))
    #     masks = masks[sort_idxs]  # Sort masks accordingly
    #     mask_weights = mask_weights[sort_idxs]

    #     # Cumulative sum of sorted masks
    #     masks_cumsum = torch.cumsum(masks, dim=0).bool().float()
        
    #     if type(start) is int:
    #         topks = torch.tensor(np.linspace(start, end, ins_steps), dtype=torch.int).to(masks.device) - 1
    #     else:
    #         topks = (torch.tensor(np.linspace(start, end, ins_steps), 
    #                             dtype=torch.float).to(masks.device) * 196).int() 
    #         topks = topks - 1

    #     masks_use = masks_cumsum[topks]

    #     masks_all.append(masks_use)
    bsz = inputs.shape[0]
    input_mask_weights_cand = attrs.to(device).float()
    input_mask_weights_cand = gaussian_blur_1d_mask(input_mask_weights_cand, kernel_size=kernel_size, sigma=sigma)  #NEW: BLUR
    num_masks = input_mask_weights_cand.shape[0] // bsz

    to_mask = attention_mask
    # if debug:
    #     import pdb; pdb.set_trace()
    # if 'token_type_ids' in kwargs:
    to_mask = to_mask * (1 - token_type_ids)
    num_ones = to_mask.sum(dim=1)

    # Mask out the parts where attention_mask is 0
    try:
        input_mask_weights_cand_masked = input_mask_weights_cand * to_mask.repeat_interleave(num_masks, dim=0)
    except:
        import pdb; pdb.set_trace()
        input_mask_weights_cand_masked = input_mask_weights_cand * to_mask.repeat_interleave(num_masks, dim=0)
    input_mask_weights_cand_float = input_mask_weights_cand.clone()

    # Sort input_mask_weights_cand_masked in descending order
    # input_mask_weights_sort_values, indices = input_mask_weights_cand_masked.sort(dim=-1, descending=True)
    # only sort top num_ones values of input_mask_weights_cand_masked and then append rest of the indices
    masks_first_indices = torch.argsort(input_mask_weights_cand_masked[:,:num_ones], dim=-1, descending=True)
    if deletion:
        masks_first_indices = masks_first_indices.flip(-1)
    indices = torch.cat([masks_first_indices,
                                       torch.arange(num_ones.item(), 
                                       input_mask_weights_cand_masked.shape[1]).to(device).unsqueeze(0)], dim=-1)
    # print('indices', indices.shape)
    # print('indices', indices)

    masks_all_all = []
    # if type(start) is int:
    #     topks = torch.tensor(np.linspace(start, end, ins_steps), dtype=torch.int).to(device) - 1
    # else:
    #     topks = (torch.tensor(np.linspace(start, end, ins_steps), 
    #                         dtype=torch.float).to(device) * 196).int() 
    #     topks = topks - 1
    if type(start) is int:
        topks = (torch.tensor(np.linspace(start, end, ins_steps), dtype=torch.float).to(device) - 1) / num_ones
    else:
        topks = torch.tensor(np.linspace(start, end, ins_steps), 
                            dtype=torch.float).to(device)

    # print('topks', topks)
    # print('attrs', attrs)
    for k in topks:
        # Create masks_all tensor initialized to zeros
        masks_all = torch.zeros_like(input_mask_weights_cand).float()

        # Calculate the number of topk elements to set to 1 for each row based on attention_mask
        
        k_values = (num_ones.float() * k).long().clamp(min=1, max=input_mask_weights_cand.shape[1])
        # print('k_values', k_values)

        # Expand k_values to match the dimensions of input_mask_weights_cand
        expanded_k_values = k_values.repeat_interleave(num_masks).view(-1, 1)
        # print('expanded_k_values', expanded_k_values)

        # Create a range tensor to compare against expanded_k_values
        range_tensor = torch.arange(input_mask_weights_cand.shape[1], 
                device=input_mask_weights_cand.device).expand_as(input_mask_weights_cand)
        # print('range_tensor', range_tensor)

        # Create the mask by comparing the range tensor with expanded_k_values
        mask = range_tensor < expanded_k_values

        # print('indices', indices.dtype)
        # print('masks_all', masks_all.dtype)
        # print('mask', mask.dtype)
        # print('mask.float()', mask.float().dtype)
        # Use the mask to set the top k values in masks_all to 1
        masks_all.scatter_(1, indices, mask.float())
        # print(masks_all)
        masks_all_all.append(masks_all)
    masks_all = torch.stack(masks_all_all, dim=0)
    # print('masks_all', masks_all.shape)
    # import pdb; pdb.set_trace()

    mask = (masks_all + token_type_ids).bool().float()


    # masks_all = torch.stack(masks_all, dim=0)
    
    # pass into the model and compute insertion scores
    mini_bsz = 32
    masks_all_shape = masks_all.shape
    masks_all_reshape = masks_all.view(-1, masks_all.shape[-1])
    # print('masks_all_reshape', masks_all_reshape.shape)

    inputs_repeat = inputs.reshape(inputs.shape[0], 1, inputs.shape[-1]) \
                    .expand(inputs.shape[0], masks_all_shape[1], inputs.shape[-1])
    
    # inputs_repeat = inputs.reshape(inputs.shape[0], 1, 3, inputs.shape[-2], inputs.shape[-1]) \
    #                 .expand(inputs.shape[0], masks_all_shape[1], 3, inputs.shape[-2], inputs.shape[-1])
    inputs_repeat_reshape = inputs_repeat.reshape(-1, inputs.shape[-1])
    attention_mask_repeat = attention_mask.reshape(attention_mask.shape[0], 1, attention_mask.shape[-1]) \
                    .expand(attention_mask.shape[0], masks_all_shape[1], attention_mask.shape[-1])
    attention_mask_repeat_reshape = attention_mask_repeat.reshape(-1, attention_mask.shape[-1])
    token_type_ids_repeat = token_type_ids.reshape(token_type_ids.shape[0], 1, token_type_ids.shape[-1]) \
                    .expand(token_type_ids.shape[0], masks_all_shape[1], 1, token_type_ids.shape[-1])
    token_type_ids_repeat_reshape = token_type_ids_repeat.reshape(-1, token_type_ids.shape[-1])
    # print('inputs_repeat_reshape', inputs_repeat_reshape.shape)
    # print('attention_mask_repeat_reshape', attention_mask_repeat_reshape.shape)
    # print('token_type_ids_repeat_reshape', token_type_ids_repeat_reshape.shape)
    
    logits = []
    for mini_bi in range(math.ceil(masks_all_reshape.shape[0] / mini_bsz)):
        # Get masked output
        inputs_mini = inputs_repeat_reshape[mini_bi * mini_bsz:(mini_bi + 1)*mini_bsz]
        attention_mask_mini = attention_mask_repeat_reshape[mini_bi * mini_bsz:(mini_bi + 1)*mini_bsz]
        token_type_ids_mini = token_type_ids_repeat_reshape[mini_bi * mini_bsz:(mini_bi + 1)*mini_bsz]

        masks_mini = masks_all_reshape[mini_bi * mini_bsz:(mini_bi + 1)*mini_bsz]
        masks_mini = (masks_mini + token_type_ids_mini).bool().float()
        # print('masks_mini', masks_mini.shape, masks_mini)
        # print(masks_mini[0])
        # print(masks_mini[-1])
        # masked_inputs = (masks_mini[:,None] * inputs_mini).long()
        # masked_attention_mask = (masks_mini[:,None] * attention_mask_mini).long()
        # masked_token_type_ids = (masks_mini[:,None] * token_type_ids_mini).long()
        
        masked_inputs = inputs_mini.clone()
        # print('masked_inputs', masked_inputs.shape)
        masked_inputs = inputs_mini.expand(masks_mini.shape).clone()
        masked_inputs[masks_mini.bool().logical_not()] = 0
        masked_attention_mask = (masks_mini[:,None] * attention_mask_mini).long()
        masked_token_type_ids = token_type_ids_mini
        # print('masked_attention_mask', masked_attention_mask.shape)
        # print('masked_token_type_ids', masked_token_type_ids.shape)
        # if explainer_name in ['xdnn']:
        #     masked_inputs = explainer.normalize(masked_inputs)
        # if explainer_name in ['bagnet']:
        #     inputs_norm = explainer.normalize(inputs_mini)
        #     masked_inputs = masks_mini[:,None] * inputs_norm
        # #     if explainer_name == 'xdnn':
        # #         import pdb; pdb.set_trace()
        # # elif explainer_name in ['bagnet']:
        # #     import pdb; pdb.set_trace()
        # elif explainer_name in ['bcos']:
        #     masked_inputs = explainer.preprocess(masked_inputs)
        #     # inputs_norm = explainer.preprocess(inputs_mini)
        #     # masked_inputs = masks_mini[:,None] * inputs_norm
        
        # print('masked_inputs', masked_inputs.shape)
        # print('masked_attention_mask', masked_attention_mask.shape)
        # print('masked_token_type_ids', masked_token_type_ids.shape)
        outputs = explainer.model(masked_inputs.squeeze(1), 
                                  attention_mask=masked_attention_mask.squeeze(1), 
                                  token_type_ids=masked_token_type_ids.squeeze(1))
        
        try:
            logits_mini = outputs.logits
        except:
            logits_mini = outputs
        logits.append(logits_mini)
    logits = torch.cat(logits, dim=0)
    logits = logits.reshape(masks_all_shape[0], masks_all_shape[1], logits.shape[-1])

    preds = logits.argmax(-1)
    probs = logits.softmax(-1)
    # print('original_preds', original_preds)
    # print('probs[range(len(preds)), :, original_preds]', probs[range(len(preds)), :, original_preds])
    # import pdb; pdb.set_trace()
    # if use_original_pred:
    return probs[range(len(preds)), :, original_preds]

def insertion_text(model, original_model, original_model_softmax, backbone_model, projection_layer,
              processor, val_config,
              method, ins_steps, start, end, 
              # use_original_pred=False, 
              debug=False, 
              deletion=False, return_all=False,
              exp_dir='/shared_data0/weiqiuy/sop/exps/multirc_bert'):
    device = backbone_model.device
    method_list = method.split('_')
    explainer_name = method_list[0]

    if val_config['dataset']['name'] == 'multirc':
        get_explainer = multirc.get_explainer
        get_dataset = multirc.get_dataset
    elif val_config['dataset']['name'] == 'movies':
        get_explainer = movies.get_explainer
        get_dataset = movies.get_dataset
    else:
        raise ValueError(f'Unknown dataset {val_config["dataset"]["name"]}')
    # if explainer_name == 'backbone':
    #     explainer_name = 'attn'
    if len(method_list) >= 2:
        suffix = f'_{"_".join(method_list[1:])}'
    else:
        suffix = ''
    
    if explainer_name in ['bagnet', 'sop']:
        ATTR_VAL_DATA_DIR = None
    else:
        ATTR_VAL_DATA_DIR = os.path.join(exp_dir, f'attributions/{explainer_name}{suffix}')

    if method == 'sop':
        explainer = model
    else:
        explainer = get_explainer(original_model, original_model_softmax, 
                                 backbone_model, processor, explainer_name, device)
    
    val_dataset, val_dataloader = get_dataset(val_config['dataset']['name'], 
                                          split=val_config['evaluation']['split'], 
                                          num_data=val_config['evaluation']['num_data'],
                                          batch_size=val_config['evaluation']['batch_size'],
                                                        # attr_dir=ATTR_VAL_DATA_DIR,
                                          processor=processor, debug=debug)

    insertion_scores = []
    insertion_scores_curve = []
    print('hi')
    for bi, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        if debug > 0 and bi >= int(debug):
            break
        elif debug < 0:
            if bi % 40 != 0:
                continue
        # if len(batch) == 5:
        #     inputs, labels, segs, attrs, idxs = batch
        # else:
        #     inputs, labels, segs, idxs = batch
        #     attrs = None
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

            attention_mask = batch['attention_mask'].to(device)
            # segs = batch['segs'].to(device).float()
        kwargs = {
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
        }
        
        
        labels = batch['label'].to(device)

        # inputs, labels, segs = inputs.to(device), labels.to(device), segs.to(device)
        with torch.no_grad():
            if explainer_name == 'sop':
                model.k = 0.2
                logits = model(inputs, kwargs=kwargs)
                original_preds = logits.argmax(-1)
                
                ins_scores_bi = []
                for k in np.linspace(start, end, ins_steps):
                    # k = int(k * 196)
                    if type(start) is int and type(end) is int and type(ins_steps) is int:
                        k = int(k)
                        k  = k / end
                    # print(" try", k, "----")

                for k in np.linspace(start, end, ins_steps):
                    # k = int(k * 196)
                    if type(start) is int and type(end) is int and type(ins_steps) is int:
                        k = int(k)
                        k  = k / end
                    # print(" hii", k, "----")
                    model.k = min(k + 0.0001, 1.)
                    # print(model.k)
                    # wrapped_backbone_model.model(inputs, output_hidden_states=True)
                    outputs = model(inputs, return_tuple=True, deletion=deletion, kwargs=kwargs)
                    logits = outputs.logits
                    preds = logits.argmax(-1)
                    probs = logits.softmax(-1)
                    ins_scores_bi.append(probs[range(len(original_preds)), preds]) #original_preds])
                    # print(ins_scores_bi)
                ins_scores_bi = torch.stack(ins_scores_bi, 0).transpose(0, 1)
                # print('ins_scores_bi', ins_scores_bi.shape)
                results = ins_scores_bi
                # print(results)
            else:
                # load saved baseline attributions
                filename = f'{bi}.pt'
                attr_filepath = os.path.join(ATTR_VAL_DATA_DIR, filename)
                attr_results = torch.load(attr_filepath)
                attributions = attr_results['expln'].attributions
                assert (attr_results['input'] == inputs).all()
                attrs = attributions

                if explainer_name in ['xdnn', 'bagnet']:
                    inputs_norm = explainer.normalize(inputs)
                    # original_logits = explainer.model(inputs_norm)
                elif explainer_name in ['bcos']:
                    inputs_norm = explainer.preprocess(inputs)
                # elif explainer_name in ['attn']: todo: for attn, pred should be new pred
                else:
                    inputs_norm = inputs

                original_logits = explainer.model(inputs_norm, **kwargs)
                if 'logits' in original_logits.__dict__:
                    original_logits = original_logits.logits
                original_preds = torch.argmax(original_logits, dim=-1)

                results = get_k_pred_text(explainer, method, inputs, kwargs, attrs, ins_steps, start, end, original_preds, 
                                     deletion=deletion, return_all=return_all)
                # print('results', results.shape, results)
                results = results.transpose(0, 1)
            insertion_scores.extend(results.mean(1).tolist())
            insertion_scores_curve.extend(results.tolist())
        
    insertion_scores_perc = []
    for i in range(len(insertion_scores_curve)):
        mean_ins = np.mean(insertion_scores_curve[i])
        max_ins = insertion_scores_curve[i][-1] # np.max(ins_scores_all_curve[i])
        insertion_scores_perc.append(mean_ins / max_ins )
        # print('mean_ins', mean_ins, 'max_ins', max_ins, 'perc', mean_ins / max_ins)
        # print('insertion_scores_curve[i]', insertion_scores_curve[i])
        # import pdb; pdb.set_trace()
        if mean_ins / max_ins == 1:
            import pdb; pdb.set_trace()
    return insertion_scores, insertion_scores_curve, insertion_scores_perc

def get_ins_del_perc_text(val_config, original_model, original_model_softmax, backbone_model, model, projection_layer, processor,
                     method, ins_steps=10, start=0.1, end=1.0, deletion=False, debug=False, vis=False,
                     exp_dir='/shared_data0/weiqiuy/sop/exps/multirc_bert'):
    ins_scores, ins_scores_curve, insertion_scores_perc = insertion_text(model, 
                                                                          original_model, 
                                                                          original_model_softmax,
                                                                          backbone_model, 
                                                                          projection_layer,
                                                                          processor, 
                                                                          val_config,
                                                                          method, 
                                             ins_steps=ins_steps, start=start, end=end, 
                                             debug=debug, return_all=True, deletion=deletion, exp_dir=exp_dir)
    if vis:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(ins_scores_original_curve[0])
        plt.show()
    print('insertion_scores_perc', np.mean(insertion_scores_perc))
    print('insertion_scores', np.mean(ins_scores))
    return {
        'scores_mean': ins_scores,
        'scores_curve': ins_scores_curve,
        'scores_perc': insertion_scores_perc,
    }