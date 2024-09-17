import math
import matplotlib.pyplot as plt

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

def get_masks_used(outputs, i=0):
    pred = outputs.logits[i].argmax(-1).item()
    pred_mask_idxs_sort = outputs.mask_weights[i,:,pred].argsort(descending=True)
    mask_weights_sort = (outputs.mask_weights * outputs.logits_all)[i,pred_mask_idxs_sort,pred]
    masks_sort = outputs.masks[i,pred_mask_idxs_sort]
    masks_sort_used = (masks_sort[mask_weights_sort != 0] > 0).int()
    mask_weights_sort_used = mask_weights_sort[mask_weights_sort != 0]
    return {
        'masks_sort_used': masks_sort_used, 
        'mask_weights_sort_used': mask_weights_sort_used
    }

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
    