from tqdm.auto import tqdm
from exlib.evaluators.common import convert_idx_masks_to_bool
import torch

# best acc
def get_acc(dataloader, explainer, method, device, k=0.1, eval_all=False):
    method_list = method.split('_')
    explainer_name = method_list[0]
    best_corrects = []
    correct = 0
    total = 0
    corrects = []
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
            
            if explainer_name == 'sop':
                explainer.k = k
                logits = explainer(inputs)
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

                masked_inputs = masks_all[:,None] * inputs
                # outputs = model(masked_inputs)
                if explainer_name in ['xdnn', 'bagnet']:
                    # inputs_norm = explainer.normalize(inputs)
                    masked_inputs = masks_mini[:,None] * inputs_norm
                elif explainer_name in ['bcos']:
                    # inputs_norm = explainer.preprocess(inputs)
                    masked_inputs = masks_mini[:,None] * inputs_norm
                outputs = explainer.model(masked_inputs)
                logits = outputs #.logits
                # -- get faithful output end --

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