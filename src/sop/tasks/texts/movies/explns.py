from ..base.explns import *
# from exlib.explainers.lime import LimeTextCls
# from exlib.explainers.shap import ShapTextCls
# from exlib.explainers.rise import RiseTextCls
# from exlib.explainers.intgrad import IntGradTextCls
# from exlib.explainers.archipelago import ArchipelagoTextCls
# from exlib.explainers.idg import IDGTextCls
# from exlib.explainers.pls import PLSTextCls
# import torch
# from exlib.modules.sop_text import gaussian_blur_1d_mask


# def get_explainer(original_model, original_model_softmax, backbone_model, processor, explainer_name, device, num_samples=20, 
#                   kernel_width=15, return_params=False):
#     # explainer_name = 'lime'
#     # original_model is a wrapped model that has .model as the actual model
#     # and the output is just the logits
#     projection_layer = original_model.model.bert.embeddings.word_embeddings

#     if explainer_name == 'lime':
#         eik = {
#             "top_labels": 2, 
#             "num_samples": num_samples
#         }

#         def split_expression(x):
#             tokens = x.split()
#             return tokens
            
#         ltek = {
#             "mask_string": "[MASK]",
#             "split_expression": split_expression
#         }

#         explainer = LimeTextCls(original_model, processor,
#                                 LimeTextExplainerKwargs=ltek,
#                                 explain_instance_kwargs=eik).to(device)
#         param_str = f'n{num_samples}'
#     elif explainer_name == 'shap':
#         sek = {'max_evals': num_samples}
#         explainer = ShapTextCls(original_model, processor, shap_explainer_kwargs=sek).to(device)
#         param_str = f'n{num_samples}'
#     elif explainer_name == 'rise':
#         if kernel_width == -1:
#             explainer = RiseTextCls(original_model_softmax, N=num_samples).to(device)
#         else:
#             import math
#             explainer = RiseTextCls(
#                 original_model_softmax, 
#                 N=num_samples,
#                 s=math.ceil(512/kernel_width)
#                 ).to(device)
#         param_str = f'n{num_samples}'
#     elif explainer_name == 'intgrad':
#         explainer = IntGradTextCls(original_model_softmax, projection_layer=projection_layer, 
#             num_steps=num_samples).to(device)
#         param_str = f'n{num_samples}'
#     elif explainer_name == 'archipelago':
#         explainer = ArchipelagoTextCls(backbone_model).to(device)
#     elif explainer_name == 'idg':
#         explainer = IDGTextCls(original_model, processor).to(device)
#     elif explainer_name == 'pls':
#         explainer = PLSTextCls(backbone_model, processor).to(device)
#     else:
#         raise ValueError('Invalid explainer name' + explainer_name)
#     explainer = explainer.to(device)

#     if return_params:
#         return explainer, param_str
#     return explainer


# def get_expln_all_classes(original_model, inputs, explainer, num_classes):
#     bsz, num_channels, H, W = inputs.shape
    
#     logits = original_model(inputs)
#     preds = torch.argmax(logits, dim=-1)

#     multi_labels = torch.tensor([list(range(num_classes))]).to(inputs.device).expand(bsz, num_classes)
    
#     expln = explainer(inputs.clone(), multi_labels.clone(), return_groups=True)

#     if 'logits' in expln._fields:
#         logits = expln.logits # use the logits of the explainer if it's faithful model

#     probs = logits[:, :num_classes].softmax(-1)
    
#     return expln, probs

# def get_attr_from_explainer(explainer, explainer_name, inputs, preds, processor, kwargs={}, device='cuda'):
#     if explainer_name in ['lime']:
#         expln = explainer(inputs, preds)
#     elif explainer_name in ['shap', 'idg']:
#         inputs_raw = [processor.decode(input_ids_i).replace('[CLS]', '').replace('[PAD]', '').strip() 
#                     for input_ids_i in inputs]

#         expln = explainer(inputs_raw, preds)
#     elif explainer_name in ['rise']:
#         expln = explainer(inputs, preds, kwargs=kwargs)
#     elif explainer_name in ['intgrad']:
#         expln = explainer(inputs, preds, x_kwargs=kwargs)
#     elif explainer_name in ['archipelago', 'pls']:
#         expln = explainer(inputs, preds, **kwargs)
#     else:
#         raise ValueError('Invalid explainer name' + explainer_name)
#     attributions = expln.attributions
#     return attributions

# def get_attr(explainer, explainer_name, batch, original_model, processor, device='cuda'):
#     # print('batch', batch)
#     if not isinstance(batch['input_ids'], torch.Tensor):
#         inputs = torch.stack(batch['input_ids']).transpose(0, 1).to(device)
#         if 'token_type_ids' in batch:
#             token_type_ids = torch.stack(batch['token_type_ids']).transpose(0, 1).to(device)
#         else:
#             token_type_ids = None
#         attention_mask = torch.stack(batch['attention_mask']).transpose(0, 1).to(device)

#         # concatenated_rows = [torch.stack(sublist) for sublist in batch['segs']]
#         # segs = torch.stack(concatenated_rows).permute(2, 0, 1).to(device).float()

#         # print('segs', segs.shape)
#     else:
#         inputs = batch['input_ids'].to(device)
#         if 'token_type_ids' in batch:
#             token_type_ids = batch['token_type_ids'].to(device)
#         else:
#             token_type_ids = None
#         attention_mask = batch['attention_mask'].to(device)

#         attention_mask = batch['attention_mask'].to(device)
#         # segs = batch['segs'].to(device).float()
#     kwargs = {
#         'token_type_ids': token_type_ids,
#         'attention_mask': attention_mask,
#     }


#     labels = batch['label'].to(device)

#     bsz = inputs.shape[0]

#     if explainer_name == 'sop':
#         inputs_dict = {
#             'inputs': inputs,
#             'kwargs': {
#                 'token_type_ids': token_type_ids,
#                 'attention_mask': attention_mask
#             }
#         }
#         outputs = explainer(return_tuple=True, **inputs_dict)
#         logits = outputs.logits
#         preds = torch.argmax(logits, dim=-1)

#         i = 0 # only for bsz 1
#         actual_nnz = ((outputs.masks[i] != 0) * kwargs['attention_mask'][i]).float().mean(0).sum() / kwargs['attention_mask'][i].sum()
#         attributions = outputs.masks
#     else:
#         inputs_dict = {
#             'input_ids': inputs,
#             'token_type_ids': token_type_ids,
#             'attention_mask': attention_mask
#         }
#         # logits = explainer.model(**inputs_dict)
#         logits = original_model(**inputs_dict)
#         preds = torch.argmax(logits, dim=-1)

#         # import pdb; pdb.set_trace()

#         # if explainer is not None:
#         # TODO make this a thing
#         attributions = get_attr_from_explainer(explainer, explainer_name, inputs, preds, processor, kwargs=kwargs, device=device)
        
#     return attributions


# def get_topk_mask(batch, attributions, k=0.2, kernel_size=1, sigma=1, add_token_type_ids=False, device='cuda'):
#     inputs = batch['input_ids']
#     attention_mask = batch['attention_mask']
#     token_type_ids = batch['token_type_ids']
#     # Calculate the number of masks
#     bsz = attention_mask.shape[0]
#     input_mask_weights_cand = attributions.to(device).float()
#     input_mask_weights_cand = gaussian_blur_1d_mask(input_mask_weights_cand, kernel_size=kernel_size, sigma=sigma)  #NEW: BLUR
#     num_masks = input_mask_weights_cand.shape[0] // bsz

#     to_mask = attention_mask
#     # if debug:
#     #     import pdb; pdb.set_trace()
#     # if 'token_type_ids' in kwargs:
#     to_mask = to_mask * (1 - token_type_ids)

#     # Mask out the parts where attention_mask is 0
#     try:
#         input_mask_weights_cand_masked = input_mask_weights_cand * to_mask.repeat_interleave(num_masks, dim=0)
#     except:
#         import pdb; pdb.set_trace()
#         input_mask_weights_cand_masked = input_mask_weights_cand * to_mask.repeat_interleave(num_masks, dim=0)
#     input_mask_weights_cand_float = input_mask_weights_cand.clone()

#     # Sort input_mask_weights_cand_masked in descending order
#     # input_mask_weights_sort_values, indices = input_mask_weights_cand_masked.sort(dim=-1, descending=True)
#     num_ones = to_mask.sum(dim=1)
#     indices = torch.cat([torch.argsort(input_mask_weights_cand_masked[:,:num_ones], 
#                            dim=-1, descending=True),
#                            torch.arange(num_ones.item(), 
#                            input_mask_weights_cand_masked.shape[1]).to(device).unsqueeze(0)], dim=-1)

#     # Create masks_all tensor initialized to zeros
#     masks_all = torch.zeros_like(input_mask_weights_cand).float()

#     # Calculate the number of topk elements to set to 1 for each row based on attention_mask
    
#     k_values = (num_ones.float() * k).long().clamp(min=1, max=input_mask_weights_cand.shape[1])

#     # Expand k_values to match the dimensions of input_mask_weights_cand
#     expanded_k_values = k_values.repeat_interleave(num_masks).view(-1, 1)

#     # Create a range tensor to compare against expanded_k_values
#     range_tensor = torch.arange(input_mask_weights_cand.shape[1], device=input_mask_weights_cand.device).expand_as(input_mask_weights_cand)

#     # Create the mask by comparing the range tensor with expanded_k_values
#     mask = range_tensor < expanded_k_values

#     # print('indices', indices.dtype)
#     # print('masks_all', masks_all.dtype)
#     # print('mask', mask.dtype)
#     # print('mask.float()', mask.float().dtype)
#     # Use the mask to set the top k values in masks_all to 1
#     masks_all.scatter_(1, indices, mask.float())

#     if add_token_type_ids:
#         # Add the token ids to the masks
#         masks_all[token_type_ids == 1] = 1
#     else:
#         masks_all[token_type_ids == 1] = 0

#     return masks_all