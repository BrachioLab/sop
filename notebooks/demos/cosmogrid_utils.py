import math
from exlib.datasets.mass_maps import map_plotter

from collections import namedtuple
import torch.nn as nn
import matplotlib.pyplot as plt


WrappedBackboneOutput = namedtuple("WrappedBackboneOutput", 
                                  ["logits",
                                   "pooler_output"])

class WrappedModel(nn.Module):
    def __init__(self, model, output_type='tuple', num_patch=14, layer=-1):
        super().__init__()
        assert output_type in ['tuple', 'logits', 'hidden_states']
        self.model = model
        self.output_type = output_type
        self.num_patch = num_patch
        self.layer = layer
    
    def forward(self, inputs):
        if self.output_type == 'tuple':
            outputs = self.model(inputs, output_hidden_states=True)
            return WrappedBackboneOutput(outputs.logits, outputs.pooler_output)
        elif self.output_type == 'logits':
            outputs = self.model(inputs)
            return outputs
        else: # hidden_states
            outputs = self.model(inputs, output_hidden_states=True)
            return outputs.hidden_states[self.layer]

def show_masked_img(img, mask, ax):
    map_plotter(img[0].cpu().numpy(), 
                    mask.cpu().numpy(),
                   ax)

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

def get_masks_used(outputs, i=0, pred=0):
    pred_mask_idxs_sort = outputs.mask_weights[i,:,pred].argsort(descending=True)
    # for mass maps, we use mask_weights c_i instead of group_attributions c_i * y_i, 
    # so we can know the percent contribution
    mask_weights_sort = outputs.mask_weights[i,pred_mask_idxs_sort,pred]
    masks_sort = outputs.masks[i,pred_mask_idxs_sort]
    masks_sort_used = (masks_sort[mask_weights_sort != 0] != 0).int()
    mask_weights_sort_used = mask_weights_sort[mask_weights_sort != 0]
    return {
        'masks_sort_used': masks_sort_used, 
        'mask_weights_sort_used': mask_weights_sort_used
    }

def show_masks_weights(inputs, expln, pred=0):
    outputs = get_masks_used(expln, i=0, pred=pred)
    masks = outputs['masks_sort_used']
    mask_weights = outputs['mask_weights_sort_used']
    denormed_img = (inputs[0:1] + 1) / 2
    print('original')
    plt.figure()
    plt.imshow(denormed_img[0].cpu().permute(1,2,0))
    plt.show()
    print('selected masks')
    titles = get_mask_weights_titles(mask_weights)
    show_masks(denormed_img[0], masks, titles=titles)
    print('all masks')
    show_masks(denormed_img[0], expln.masks[0], cols=14)
    