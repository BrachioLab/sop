from tqdm.auto import tqdm
from exlib.evaluators.common import convert_idx_masks_to_bool
import torch
from ..utils.metric_utils import get_prob_obj, get_prob_obj_coverage

# best acc
def get_acc_purity(dataloader, explainer, method, device, k=0.1, eval_all=False, built_in=False, num_classes=1000, num_groups=20):
    """
    If built_in is True, then for built-in explainers, we use the logits from the explainer.
    """
    method_list = method.split('_')
    explainer_name = method_list[0]
    best_corrects = []
    correct = 0
    total = 0
    corrects = []
    entropies = []
    ratios_obj_coverage = []
    ratios_obj = []

    if built_in:
        explainer_names_builtin = ['sop', 'bagnet', 'backbone', 'xdnn']
    else:
        explainer_names_builtin = ['sop']
    with torch.no_grad():
        progress_bar_eval = tqdm(range(len(dataloader)))
        for bi, batch in enumerate(dataloader):
            if not eval_all:
                if bi % 10 != 0:
                    progress_bar_eval.update(1)
                    continue
            # Now you can use `inputs` and `labels` in your training loop.
            # print(len(batch))
            if len(batch) == 5:
                inputs, labels, segs, attrs, idxs = batch
            else:
                inputs, labels, segs, idxs = batch
                attrs = None
            # inputs, labels, segs = inputs.to(device), labels.to(device), segs.to(device)
            # inputs, labels, attrs = batch
            inputs, labels = inputs.to(device), labels.to(device)
            if attrs is not None:
                attrs = attrs.to(device)
            
            if explainer_name in explainer_names_builtin:
                if explainer_name in ['sop', 'backbone']:
                    try:
                        explainer.k = k
                    except:
                        pass
                    logits = explainer(inputs)
                    if not isinstance(logits, torch.Tensor):
                        logits = logits.logits
                elif explainer_name == 'bagnet':
                    # use top 20 groups
                    bsz = inputs.shape[0]
                    multi_labels = torch.tensor([list(range(num_classes))]).to(inputs.device).expand(bsz, num_classes)
                    expln = explainer(inputs.clone(), multi_labels.clone(), return_groups=True)
                    mask_weights = expln.group_attributions.flatten(1, 2)
                    sort_idxs = torch.argsort(mask_weights, dim=1, descending=True)
                    sorted_mask_weights = torch.gather(mask_weights, dim=1, index=sort_idxs)
                    logits = sorted_mask_weights[:,:num_groups].sum(1) # take top 20 masks
                elif explainer_name == 'xdnn':
                    bsz = inputs.shape[0]
                    multi_labels = torch.tensor([list(range(num_classes))]).to(inputs.device).expand(bsz, num_classes)
                    expln = explainer(inputs.clone(), multi_labels.clone(), return_groups=True)
                    mask_weights = expln.attributions.mean(1).flatten(1, 2)
                    topk = int(mask_weights.shape[1] * k)
                    topk_mask_weights = torch.topk(mask_weights, topk, dim=1).values
                    logits = topk_mask_weights.sum(1)
                else:
                    raise ValueError(f'Unsupported explainer: {explainer_name}')
            else:
                # get_faithful_output
                if explainer_name in ['xdnn', 'bagnet']:
                    inputs_norm = explainer.normalize(inputs)
                    # masked_inputs = masks_mini[:,None] * inputs_norm
                elif explainer_name in ['bcos']:
                    inputs_norm = explainer.preprocess(inputs)
                    # masked_inputs = masks_mini[:,None] * inputs_norm
                else:
                    inputs_norm = inputs

                with torch.no_grad():
                    original_logits = explainer.model(inputs_norm)
                if not isinstance(original_logits, torch.Tensor):
                    original_logits = original_logits.logits
                preds = torch.argmax(original_logits, dim=-1)

                masks_all = []
                for idx in range(len(inputs)):
                    # Create a mask of size (28, 28) with values from 1 to 28*28
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

                        masks = convert_idx_masks_to_bool(resized_mask[None]).to(preds.device)
                        mask_weights = (masks.to(preds.device) * attrs[idx][0:1].to(preds.device)).sum(-1).sum(-1).to(preds.device)
                        sort_idxs = torch.argsort(mask_weights).flip(-1)
                        masks = masks[sort_idxs]
                        mask_weights = mask_weights[sort_idxs]

                    topk = int(masks.shape[0] * k)
                    masks_use = masks[:topk]
                    mask = masks_use.sum(0).bool().float()

                    masks_all.append(mask)

                masks_all = torch.stack(masks_all, dim=0)

                masked_inputs = masks_all[:,None] * inputs_norm
                if explainer_name in ['xdnn']: #, 'bagnet']:
                    masked_inputs = masks_all[:,None] * inputs
                    masked_inputs = explainer.normalize(masked_inputs)
                elif explainer_name in ['bagnet']:
                    inputs_norm = explainer.normalize(inputs)
                    masked_inputs = masks_all[:,None] * inputs_norm
                elif explainer_name in ['bcos']:
                    masked_inputs = masks_all[:,None] * inputs
                    masked_inputs = explainer.preprocess(masked_inputs)
                outputs = explainer.model(masked_inputs)
                if not isinstance(outputs, torch.Tensor):
                    logits = outputs.logits
                else:
                    logits = outputs #.logits

            preds = torch.argmax(logits, dim=-1)

            bsz = inputs.shape[0]
            correct += (preds == labels).sum().item()
            corrects.extend((preds == labels).view(-1).cpu().numpy().tolist())
            total += bsz
            progress_bar_eval.update(1)
    return {
        'acc': correct / total,
        'corrects': corrects
    }