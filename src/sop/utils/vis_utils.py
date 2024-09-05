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