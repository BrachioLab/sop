import math
import matplotlib.pyplot as plt

from IPython.display import Markdown, display


def get_masks_used(outputs, i=0, debug=False, use_mask_weights_only=False):
    if debug:
        import pdb; pdb.set_trace()
    pred = outputs.logits[i].argmax(-1).item()
    pred_mask_idxs_sort = outputs.mask_weights[i,:,pred].argsort(descending=True)
    if use_mask_weights_only:
        mask_weights_sort = outputs.mask_weights[i,pred_mask_idxs_sort,pred]
        # print('b mask_weights_sort', mask_weights_sort.shape)
        # import pdb; pdb.set_trace()
    else:
        mask_weights_sort = (outputs.mask_weights * outputs.logits_all)[i,pred_mask_idxs_sort,pred]
        # print('mask_weights_sort', mask_weights_sort.shape)
    
    masks_sort = outputs.masks[i,pred_mask_idxs_sort]
    masks_sort_used = (masks_sort[mask_weights_sort != 0] > 0).int()
    mask_weights_sort_used = mask_weights_sort[mask_weights_sort != 0]
    if debug:
        import pdb; pdb.set_trace()
    return {
        'masks_sort_used': masks_sort_used, 
        'mask_weights_sort_used': mask_weights_sort_used
    }

# img vis
def show_masked_img(img, mask, ax):
    ax.imshow(img.permute(1,2,0).cpu().numpy())
    ax.imshow(mask.cpu().numpy(), cmap='hot', alpha=0.5)
    ax.contour(mask.cpu().numpy(), 2, colors='black', linestyles='dashed')
    ax.contourf(mask.cpu().numpy(), 2, hatches=['//', None, None],
                cmap='gray', extend='neither', linestyles='-', alpha=0.01)
    ax.axis('off')
    
def show_masks(img, masks, titles=None, cols=5, imgsize=3):
    n_masks = len(masks)
    rows = math.ceil(n_masks / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * imgsize, rows * imgsize))
    axes = axes.ravel()
    for m_i in range(len(masks)):
        mask = masks[m_i]
        show_masked_img(img, mask, axes[m_i])
        if titles is not None:
            axes[m_i].set_title(titles[m_i])
    for m_i in range(len(masks), len(axes)):
        axes[m_i].axis('off')
    plt.show()
    
def get_mask_weights_titles(mask_weights):
    titles = [f'{mask_weight.item():.10f}' for mask_weight in mask_weights]
    return titles

def show_masks_weights(inputs, expln, i=0):
    outputs = get_masks_used(expln, i=i)
    masks = outputs['masks_sort_used']
    mask_weights = outputs['mask_weights_sort_used']
    denormed_img = (inputs[i:i+1] + 1) / 2
    print('original')
    plt.figure()
    plt.imshow(denormed_img[0].cpu().permute(1,2,0))
    plt.show()
    print('selected masks')
    titles = get_mask_weights_titles(mask_weights)
    show_masks(denormed_img[0], masks, titles=titles)
    print('all masks')
    show_masks(denormed_img[0], expln.masks[i], cols=14)
   
# Text vis
def printmd(string):
    display(Markdown(string))

def show_attrs(tokens, masks):
    print_str = ""
    for i in range(len(tokens)):
        expln_val = masks[i].item()
        if expln_val == 0:
            print_str += f" {tokens[i]}"
        else:
            print_str += f"<span style='background-color:rgb({255 * (1-expln_val)},{255 * expln_val},{255 * expln_val})'> {tokens[i]}</span>"
    return print_str

def show_masked_text(inputs, mask, processor, idx=0):
    tokens = processor.convert_ids_to_tokens(inputs[idx])
    printmd(show_attrs(tokens, mask[idx]))
