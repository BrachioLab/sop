import torch.nn as nn
from tqdm.auto import tqdm
from exlib.modules.sop_text import gaussian_blur_1d_mask
import os
import torch
from ..tasks.texts.base.explns import get_attr_from_explainer
from ..tasks.texts.multirc import get_topk_mask, get_attr


def get_acc_text(
    explainer,
    original_model,
    # model, original_model, original_model_softmax, 
            # backbone_model, 
            processor, dataloader, explainer_name, 
            suffix='', kernel_size=1, sigma=1, from_save=True, k=0.2, 
            exp_dir='/shared_data0/weiqiuy/sop/exps/multirc_bert/',
            device='cuda', debug=True):
    # model = wrapped_backbone_model
    criterion = nn.CrossEntropyLoss()
    
    
    # explainer = None
    
    # if explainer_name != 'sop' and not from_save:
    #     explainer = sop.tasks.multirc.get_explainer(original_model, original_model_softmax, 
    #                                             backbone_model, processor, explainer_name, device)

    attr_dir = os.path.join(exp_dir, f'attributions/{explainer_name}{suffix}')

    not_exist = []
    corrects = []

    with torch.no_grad():
        total_loss = 0.0
        correct = 0
        total = 0
        total_nnz = 0
        total_num_masks = 0
        total_coverage = 0
        total_nnz_all = 0
        correct_max = 0
        correct_num = 0
        progress_bar_eval = tqdm(range(len(dataloader)))
        for bi, batch in enumerate(dataloader):
            if debug:
                if bi % 10 != 0:
                    progress_bar_eval.update(1)
                    continue
                if bi > 1000:
                    break
            # Now you can use `inputs` and `labels` in your training loop.
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
            
            bsz = inputs.shape[0]
            
            if explainer_name == 'sop':
                inputs_dict = {
                    'inputs': inputs,
                    'kwargs': {
                        'token_type_ids': token_type_ids,
                        'attention_mask': attention_mask
                    }
                }
                explainer.k = k
                outputs = explainer(return_tuple=True, **inputs_dict)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                
                i = 0 # only for bsz 1
                actual_nnz = ((outputs.masks[i] != 0) * kwargs['attention_mask'][i]).float().mean(0).sum() / kwargs['attention_mask'][i].sum()
                    
            else:
                inputs_dict = {
                    'input_ids': inputs,
                    'token_type_ids': token_type_ids,
                    'attention_mask': attention_mask
                }
                logits = original_model(**inputs_dict)
                preds = torch.argmax(logits, dim=-1)
    
                # import pdb; pdb.set_trace()

                # if explainer is not None:
                if not from_save:
                    expln = get_attr_from_explainer(explainer, explainer_name, inputs, preds, processor, 
                        kwargs=kwargs, device=device, return_expln=True)
                    # if explainer_name in ['lime']:
                    #     expln = explainer(inputs, preds)
                    # elif explainer_name in ['shap', 'idg']:
                    #     inputs_raw = [processor.decode(input_ids_i).replace('[CLS]', '').replace('[PAD]', '').strip() 
                    #                 for input_ids_i in inputs]

                    #     expln = explainer(inputs_raw, preds)
                    # elif explainer_name in ['rise']:
                    #     expln = explainer(inputs, preds, kwargs=kwargs)
                    # elif explainer_name in ['intgrad']:
                    #     expln = explainer(inputs, preds, x_kwargs=kwargs)
                    # elif explainer_name in ['archipelago', 'pls']:
                    #     expln = explainer(inputs, preds, **kwargs)
                    # else:
                    #     raise ValueError('Invalid explainer name' + explainer_name)
                    attributions = expln.attributions
                else:
                    filename = f'{bi}.pt'
                    attr_filepath = os.path.join(attr_dir, filename)
                    if not os.path.exists(attr_filepath):
                        not_exist.append(attr_filepath)
                        # print(f'File {attr_filepath} does not exist')
                        continue
                    attr_results = torch.load(attr_filepath)
                    attributions = attr_results['expln'].attributions
                    if attributions.shape[1] == 1:
                        attributions = get_attr(explainer, explainer_name, batch, original_model, 
                                processor, device=device)
                    assert (attr_results['input'] == inputs).all()

                # Calculate the number of masks
                bsz = attention_mask.shape[0]
                input_mask_weights_cand = attributions.to(device).float()
                input_mask_weights_cand = gaussian_blur_1d_mask(input_mask_weights_cand, kernel_size=kernel_size, sigma=sigma)  #NEW: BLUR
                num_masks = input_mask_weights_cand.shape[0] // bsz

                to_mask = attention_mask
                # if debug:
                #     import pdb; pdb.set_trace()
                # if 'token_type_ids' in kwargs:
                to_mask = to_mask * (1 - token_type_ids)

                # Mask out the parts where attention_mask is 0
                try:
                    input_mask_weights_cand_masked = input_mask_weights_cand * to_mask.repeat_interleave(num_masks, dim=0)
                except:
                    import pdb; pdb.set_trace()
                    input_mask_weights_cand_masked = input_mask_weights_cand * to_mask.repeat_interleave(num_masks, dim=0)
                input_mask_weights_cand_float = input_mask_weights_cand.clone()

                num_ones = to_mask.sum(dim=1)
                # Sort input_mask_weights_cand_masked in descending order
                # input_mask_weights_sort_values, indices = input_mask_weights_cand_masked.sort(dim=-1, descending=True)
                # indices = torch.cat([torch.argsort(input_mask_weights_cand_masked[:,:num_ones], 
                #                        dim=-1, descending=True),
                #                        torch.arange(num_ones.item(), 
                #                        input_mask_weights_cand_masked.shape[1]).to(device).unsqueeze(0)], dim=-1)
                masks_first_indices = torch.argsort(input_mask_weights_cand_masked[:,:num_ones], dim=-1, descending=True)
                
                indices = torch.cat([masks_first_indices,
                                                torch.arange(num_ones.item(), 
                                                input_mask_weights_cand_masked.shape[1]).to(device).unsqueeze(0)], dim=-1)
                

                # Create masks_all tensor initialized to zeros
                masks_all = torch.zeros_like(input_mask_weights_cand).float()

                # Calculate the number of topk elements to set to 1 for each row based on attention_mask
                k_values = (num_ones.float() * k).long().clamp(min=1, max=input_mask_weights_cand.shape[1])

                # Expand k_values to match the dimensions of input_mask_weights_cand
                expanded_k_values = k_values.repeat_interleave(num_masks).view(-1, 1)

                # Create a range tensor to compare against expanded_k_values
                range_tensor = torch.arange(input_mask_weights_cand.shape[1], device=input_mask_weights_cand.device).expand_as(input_mask_weights_cand)

                # Create the mask by comparing the range tensor with expanded_k_values
                mask = range_tensor < expanded_k_values

                # print('indices', indices.dtype)
                # print('masks_all', masks_all.dtype)
                # print('mask', mask.dtype)
                # print('mask.float()', mask.float().dtype)
                # Use the mask to set the top k values in masks_all to 1
                masks_all.scatter_(1, indices, mask.float())


                mask = (masks_all + token_type_ids).bool().float()

                # print("attr_results['input']", attr_results['input'].shape)
                # print("(attr_results['input'] * mask).long()[None]", (attr_results['input'] * mask).long()[None].shape)
                # print("(attr_results['attention_mask']* mask).long()[None]", (attr_results['attention_mask']* mask).long()[None].shape)
                # print("attr_results['token_type_ids'][None]", attr_results['token_type_ids'][None].shape)
                outputs = explainer.model((inputs * mask).long(),
                               attention_mask=(attention_mask* mask).long(),
                               token_type_ids = token_type_ids.long())
                if isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logits = outputs.logits
                    
                preds = logits.argmax(-1)
            
                # nnz
                # import pdb; pdb.set_trace()
                actual_nnz = (to_mask * mask).sum() / num_ones
                # actual_nnz = (attention_mask* mask).sum() / (attention_mask* mask).shape[-1]
            total_nnz += actual_nnz.item()
            
            correct += (preds[0] == labels).item()
            corrects.extend((preds == labels).view(-1).cpu().numpy().tolist())
            total += 1
            progress_bar_eval.update(1)
    print('number of not exist files', len(not_exist))
    acc = correct / total
    print(explainer_name, acc)
    result = {
        'val_acc': acc,
        'val_nnz': total_nnz / total,
        'acc': acc,
        'corrects': corrects
    }
    return result