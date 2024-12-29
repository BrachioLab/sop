import torch
from ..tasks.texts.base.explns import get_explainer, get_attr_from_explainer, get_topk_mask
from ..utils.vis_utils import show_masked_text

def show_some_text_egs(val_dataloader, model, original_model, original_model_softmax, backbone_model, 
                       processor, explainer_name, device='cuda', num_examples=10):
    batch = next(iter(val_dataloader))
    for bi, batch in enumerate(val_dataloader):
        # print('len(val_dataloader) // 10', len(val_dataloader) // 10)
        if bi % (len(val_dataloader) // num_examples) != 0:
            continue
        # print('bi', bi)
        # print(bi % (len(val_dataloader) // 10) != 0)
        # print(bi % (len(val_dataloader) // 10))
        # if bi > 50:
        #     break
        if not isinstance(batch['input_ids'], torch.Tensor):
            inputs = torch.stack(batch['input_ids']).transpose(0, 1).to(device)
            if 'token_type_ids' in batch:
                token_type_ids = torch.stack(batch['token_type_ids']).transpose(0, 1).to(device)
            else:
                token_type_ids = None
            attention_mask = torch.stack(batch['attention_mask']).transpose(0, 1).to(device)
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

        if explainer_name == 'sop':
            inputs_dict = {
                'inputs': inputs,
                'kwargs': {
                    'token_type_ids': token_type_ids,
                    'attention_mask': attention_mask
                }
            }
            outputs = model(return_tuple=True, **inputs_dict)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            from sop.utils.vis_utils import get_masks_used
            outputs = get_masks_used(outputs, i=0)
            masks = outputs['masks_sort_used']
            mask_weights = outputs['mask_weights_sort_used']
            # import pdb; pdb.set_trace()
            mask = masks[0:1]
            
            
        else:
            inputs_dict = {
                'input_ids': inputs,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask
            }
            logits = original_model(**inputs_dict)
            preds = torch.argmax(logits, dim=-1)

            explainer = get_explainer(original_model, original_model_softmax, backbone_model, processor, 
                              explainer_name, device, num_samples=20, 
                              kernel_width=15)
            expln = get_attr_from_explainer(explainer, explainer_name, inputs, preds, processor, 
                                    kwargs=kwargs, device=device, return_expln=True)

            mask = get_topk_mask(inputs_dict, expln.attributions, k=0.2, kernel_size=1, 
                                 sigma=1, add_token_type_ids=True, device=device)
        show_masked_text(inputs, mask, processor, idx=0)
        print('preds', preds[0])
        print('labels', labels[0])
        

# explainer_name = 'sop'
# show_some_text_egs(val_dataloader, model, original_model, original_model_softmax, backbone_model, 
#                    processor, explainer_name, device=device)