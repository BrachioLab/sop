import torch.nn as nn
from tqdm.auto import tqdm
from exlib.modules.sop_text import gaussian_blur_1d_mask
import os
import torch
from collections import defaultdict
from tqdm.auto import tqdm


from ..tasks.texts.multirc import get_topk_mask, get_attr
from ..utils.text_utils import map_token_spans_to_original_text, find_start_end_tokens
from ..utils.text_utils import find_evidence_indices_list
from .utils import get_entropy_text, get_prob_obj_text, get_prob_obj_coverage_text, get_iou_text
from ..utils.vis_utils import get_masks_used


def create_binary_mask(spans, text_length):
    # Create an array of zeros with length equal to the text length
    mask = [0] * text_length
    
    # Set the indices within each span to 1
    for start, end in spans:
        for index in range(start, end + 1):
            mask[index] = 1
    
    return mask

def get_purity_results(explainer, explainer_name, batch, raw_batch, original_model, 
        tokenizer, verbose=0, device='cuda', best_only=False, idx=None,
        exp_dir='/shared_data0/weiqiuy/sop/exps/multirc_bert/', from_save=True, suffix=''):
    """
    each value in batch is size (N, L)
    """
    if idx is not None:
        attr_dir = os.path.join(exp_dir, f'attributions/{explainer_name}{suffix}')
        filename = f'{idx}.pt'
        attr_filepath = os.path.join(attr_dir, filename)
        # print('idx', idx)
        # attr = get_attr(explainer, explainer_name, batch, original_model, tokenizer, device=device)
        if os.path.exists(attr_filepath):
            attr = torch.load(attr_filepath)['expln'].attributions
            # print('attr', attr.shape, 'idx', idx)
            if attr.shape[1] == 1:
                attr = get_attr(explainer, explainer_name, batch, original_model, tokenizer, device=device)
                # print('attr new', attr.shape)
                # import pdb; pdb.set_trace()
        else:
            attr = get_attr(explainer, explainer_name, batch, original_model, tokenizer, device=device)
            # os.makedirs(attr_dir, exist_ok=True)
            # torch.save(attr, attr_filepath)
    else:
        attr = get_attr(explainer, explainer_name, batch, original_model, tokenizer, device=device)

    # get gold evidence indicies
    # Example usage:
    # passage = "This is the sentence where we find the specific evidence."
    # evidence = "specific evidence"
    passage = raw_batch['passage']
    evidences = raw_batch['evidences']
    indices = find_evidence_indices_list(passage, evidences)

    # get span indices (on the original split text) of pred evidence
    if explainer_name == 'sop':
        # bin_mask = attr[0]
        inputs = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        inputs_dict = {
                'inputs': inputs,
                'kwargs': {
                    'token_type_ids': token_type_ids,
                    'attention_mask': attention_mask
                }
            }
        outputs = explainer(return_tuple=True, **inputs_dict)
        
        used_outputs = get_masks_used(outputs, i=0)
        masks = used_outputs['masks_sort_used']
        mask_weights = used_outputs['mask_weights_sort_used']
        if best_only:
            # print('masks', masks.shape)
            try:
                bin_mask = masks[0][None]
            except:
                # import pdb; pdb.set_trace()
                used_outputs = get_masks_used(outputs, i=0, use_mask_weights_only=True)
                masks = used_outputs['masks_sort_used']
                mask_weights = used_outputs['mask_weights_sort_used']
                bin_mask = masks[0][None]
        else:
            bin_mask = masks # use all used masks for now
        # print('bin_mask', bin_mask.shape)
    else:
        bin_mask = get_topk_mask(batch, attr, device=device)
    
    results_all = []
    for i in range(bin_mask.shape[0]):
        spans = find_start_end_tokens(bin_mask.cpu()[i])
        # print('spans', spans)
        output = map_token_spans_to_original_text(raw_batch['passage'], spans, tokenizer)
        # outputs.append(output)
    # spans = find_start_end_tokens(bin_mask.cpu()[0])
    # output = map_token_spans_to_original_text(raw_batch['passage'], spans, tokenizer)
    
    
    
        # get binary gold mask from span indices
        text_length = len(output.original_text_tokens)
        gold_mask = create_binary_mask(indices, text_length)
        if verbose >= 1:
            print(gold_mask)
        
        # get binary pred mask from span indices
        pred_mask = create_binary_mask(output.original_span_indices, text_length)
        if verbose >= 1:
            print(pred_mask)
        
        # get all metrics
        pred_mask_pt = torch.tensor(pred_mask)[None].to(device)
        gold_mask_pt = torch.tensor(gold_mask)[None].to(device)
        entropy = get_entropy_text(pred_mask_pt, gold_mask_pt)
        prob_obj = get_prob_obj_text(pred_mask_pt, gold_mask_pt)
        prob_obj_coverage = get_prob_obj_coverage_text(pred_mask_pt, gold_mask_pt)
        iou = get_iou_text(pred_mask_pt, gold_mask_pt)

        results = {
            'entropy': entropy, 
            'prob_obj': prob_obj, 
            'prob_obj_coverage': prob_obj_coverage, 
            'iou': iou,
            'purity': 1 - entropy
        }
        results_all.append(results)
    
    return results_all
    

def get_acc_purity_text(explainer, explainer_name, original_model, tokenizer, val_dataset, 
        val_dataset_raw, device='cuda', debug=False, best_only=False, from_save=True, suffix='', 
        exp_dir='/shared_data0/weiqiuy/sop/exps/multirc_bert/'):
    purity_results_all = defaultdict(list)
    for idx in tqdm(range(len(val_dataset))):
        if debug:
            if idx % 10 != 0:
                continue
            # if idx < 650:
            #     continue
            if idx > 1000:
                break
        batch = val_dataset[idx]
        batch = {k: torch.tensor(v)[None].to(device) for k, v in batch.items()}
        raw_batch = val_dataset_raw[idx]
        purity_results_list = get_purity_results(explainer, explainer_name, batch, raw_batch, original_model, 
                                                 tokenizer, device=device, best_only=best_only, idx=idx, 
                                                 from_save=from_save, exp_dir=exp_dir, suffix=suffix)
        # purity_results
        for purity_results in purity_results_list:
            for k, v in purity_results.items():
                purity_results_all[k].append(v)
    return purity_results_all
