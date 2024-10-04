import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
from functools import partial


from collections import namedtuple

WrappedBackboneOutput = namedtuple("WrappedBackboneOutput", 
                                  ["logits",
                                   "pooler_output"])


class WrappedModel(nn.Module):
    def __init__(self, model, output_type='tuple', num_patch=14):
        super().__init__()
        assert output_type in ['tuple', 'logits', 'hidden_states']
        self.model = model
        self.output_type = output_type
        self.num_patch = num_patch
    
    def forward(self, inputs):
        if self.output_type == 'tuple':
            outputs = self.model(inputs, output_hidden_states=True)
            return WrappedBackboneOutput(outputs.logits, outputs.hidden_states[-1][:,0])
        elif self.output_type == 'logits':
            outputs = self.model(inputs)
            return outputs.logits
        else: # hidden_states
            # import pdb; pdb.set_trace()
            outputs = self.model(inputs, output_hidden_states=True)
            hidden_size = outputs.hidden_states[-2].shape[-1]
            return outputs.hidden_states[-2][:,1:].transpose(1,2).reshape(-1, 
                            hidden_size, self.num_patch, self.num_patch)

        
def get_chained_attr(obj, attr_chain):
    attrs = attr_chain.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def get_wrapped_models(model, config):
    wrapped_model = WrappedModel(model, output_type='tuple')
    class_weight_layer = config.class_weight_layer if 'class_weight_layer' in config.__dict__ and config.class_weight_layer is not None else config.finetune_layers[0]
    class_weights = get_chained_attr(wrapped_model, class_weight_layer).weight #.clone().to(device)
    # projection_layer = WrappedModel(model, output_type='hidden_states')
    projection_layer = WrappedModel(wrapped_model, output_type='hidden_states')
    return wrapped_model, class_weights, projection_layer


def convert_idx_masks_to_bool(masks):
    """
    input: masks (1, img_dim1, img_dim2)
    output: masks_bool (num_masks, img_dim1, img_dim2)
    """
    unique_idxs = torch.sort(torch.unique(masks)).values
    idxs = unique_idxs.view(-1, 1, 1)
    broadcasted_masks = masks.expand(unique_idxs.shape[0], 
                                     masks.shape[1], 
                                     masks.shape[2])
    masks_bool = (broadcasted_masks == idxs)
    return masks_bool


def convert_idx_masks_to_bool_big_first(masks):
    """
    input: masks (1, img_dim1, img_dim2)
    output: masks_bool (num_masks, img_dim1, img_dim2)
    """
    unique_idxs, counts = torch.unique(masks, return_counts=True)
    unique_idxs_sorted = unique_idxs[counts.argsort(descending=True)]
    idxs = unique_idxs_sorted.view(-1, 1, 1)
    broadcasted_masks = masks.expand(unique_idxs.shape[0], 
                                     masks.shape[1], 
                                     masks.shape[2])
    masks_bool = (broadcasted_masks == idxs)
    return masks_bool


def get_mask_transform(num_masks_max=200, processor=None, big_first=False):
    def mask_transform(mask):
        seg_mask_cut_off = num_masks_max
        # Preprocess the mask using the ViTImageProcessor
        if len(mask.shape) == 2 and mask.dtype == torch.bool:
            mask_dim1, mask_dim2 = mask.shape
            mask = mask.unsqueeze(0).expand(3, 
                                            mask_dim1, 
                                            mask_dim2).float()
            if processor is not None:
                inputs = processor(mask, 
                                do_rescale=False, 
                                do_normalize=False,
                                return_tensors='pt')
                # (1, 3, 224, 224)
                return inputs['pixel_values'][0][0]
            else:
                return mask
        else: # len(mask.shape) == 3
            if mask.dtype != torch.bool:
                if len(mask.shape) == 2:
                    mask = mask.unsqueeze(0)
                if big_first:
                    mask = convert_idx_masks_to_bool_big_first(mask)
                else:
                    mask = convert_idx_masks_to_bool(mask)
            bsz, mask_dim1, mask_dim2 = mask.shape
            mask = mask.unsqueeze(1).expand(bsz, 
                                            3, 
                                            mask_dim1, 
                                            mask_dim2).float()

            if bsz < seg_mask_cut_off:
                repeat_count = seg_mask_cut_off // bsz + 1
                mask = torch.cat([mask] * repeat_count, dim=0)

            # add additional mask afterwards
            mask_sum = torch.sum(mask[:seg_mask_cut_off - 1], dim=0, keepdim=True).bool()
            if False in mask_sum:
                mask = mask[:seg_mask_cut_off - 1]
                compensation_mask = (1 - mask_sum.int()).bool()
                mask = torch.cat([mask, compensation_mask])
            else:
                mask = mask[:seg_mask_cut_off]

            if processor is not None:
                inputs = processor(mask, 
                                do_rescale=False, 
                                do_normalize=False,
                                return_tensors='pt')
                
                return inputs['pixel_values'][:,0]
            else:
                return mask[:,0]
    return mask_transform


def compress_single_masks(masks, masks_weights, min_size):
    # num_masks, seq_len = masks.shape
    masks_bool = (masks > 0).int()
    sorted_weights, sorted_indices = torch.sort(masks_weights, descending=True)
    sorted_indices = sorted_indices[sorted_weights > 0]

    
    try:
        masks_bool = masks_bool[sorted_indices]  # sorted masks
    except:
        import pdb; pdb.set_trace()
        masks_bool = masks_bool[sorted_indices]  # sorted masks
    
    masks = torch.zeros(*masks_bool.shape[1:]).to(masks.device)
    count = 1
    for mask in masks_bool:
        new_mask = mask.bool() ^ (mask.bool() & masks.bool())
        if torch.sum(new_mask) >= min_size:
            masks[new_mask] = count
            count += 1

    masks = masks - 1
    masks = masks.int()
    masks[masks == -1] = torch.max(masks) + 1

    return masks

def compress_masks(masks, masks_weights, min_size=0):
    new_masks = []
    for i in range(len(masks)):
        compressed_mask = compress_single_masks(masks[i], masks_weights[i], 
                                                min_size)
        new_masks.append(compressed_mask)
    return torch.stack(new_masks)


def compress_masks_image(masks, masks_weights, min_size=0):
    assert len(masks.shape) == 4 # bsz, num_masks, img_dim_1, img_dim_2 = masks.shape
    return compress_masks(masks, masks_weights, min_size)
    

def compress_masks_text(masks, masks_weights, min_size=0):
    assert len(masks.shape) == 3 # bsz, num_masks, seq_len = masks.shape
    return compress_masks(masks, masks_weights, min_size)
           

def _get_inverse_sqrt_with_separate_heads_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, 
    num_steps_per_epoch: int,
    timescale: int = None, 
    num_heads: int = 1, 
):
    epoch = current_step // (num_steps_per_epoch * num_heads)
    steps_within_epoch = current_step % num_steps_per_epoch
    step_for_curr_head = epoch * num_steps_per_epoch + steps_within_epoch
    if step_for_curr_head < num_warmup_steps:
        return float(step_for_curr_head) / float(max(1, num_warmup_steps))
    shift = timescale - num_warmup_steps
    decay = 1.0 / math.sqrt((step_for_curr_head + shift) / timescale)
    return decay

def get_inverse_sqrt_with_separate_heads_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_steps_per_epoch: int,
    timescale: int = None, 
    num_heads: int = 1, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    if timescale is None:
        timescale = num_warmup_steps

    lr_lambda = partial(
        _get_inverse_sqrt_with_separate_heads_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_steps_per_epoch=num_steps_per_epoch,
        timescale=timescale,
        num_heads=num_heads,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


"""Sparsemax activation function.

Pytorch implementation of Sparsemax function from:
-- "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
-- André F. T. Martins, Ramón Fernandez Astudillo (http://arxiv.org/abs/1602.02068)
"""


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation
        
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, inputs):
        """Forward function.

        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size

        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor

        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        device = inputs.device
        inputs = inputs.transpose(0, self.dim)
        original_size = inputs.size()
        inputs = inputs.reshape(inputs.size(0), -1)
        inputs = inputs.transpose(0, 1)
        dim = 1

        number_of_logits = inputs.size(dim)

        # Translate input by max for numerical stability
        inputs = inputs - torch.max(inputs, dim=dim, keepdim=True)[0].expand_as(inputs)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=inputs, dim=dim, descending=True)[0]
        range_tensor = torch.arange(start=1, end=number_of_logits + 1, step=1, 
                                    device=device, dtype=inputs.dtype).view(1, -1)
        range_tensor = range_tensor.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range_tensor * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(inputs.type())
        k = torch.max(is_gt * range_tensor, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(inputs)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(inputs), inputs - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input


def same_tensor(tensor, *args):
    ''' Do the input tensors all point to the same underlying data '''
    for other in args:
        if not torch.is_tensor(other):
            return False

        if tensor.device != other.device:
            return False

        if tensor.dtype != other.dtype:
            return False

        if tensor.data_ptr() != other.data_ptr():
            return False

    return True


class SparseMultiHeadedAttention(nn.Module):
    ''' 
    Implement a sparse multi-headed attention module. 
    Code adapted from multi-headed attention from https://github.com/dojoteef/synst/blob/master/models/attention.py 
    '''
    def __init__(self, embed_dim, num_heads=1, scale=1):
        ''' Initialize the attention module '''
        super().__init__()

        # ensure valid inputs
        assert embed_dim % num_heads == 0, \
            f'num_heads={num_heads} should evenly divide embed_dim={embed_dim}'

        # store off the scale and input params
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.scale = scale
        # self.scale = self.projection_dim ** -0.5

        # Combine projections for multiple heads into a single linear layer for efficiency
        self.input_weights = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.output_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.sparsemax = Sparsemax(dim=-1)
        self.reset_parameters()

    def reset_parameters(self):
        ''' Reset parameters using xavier initialization '''
        # Initialize using Xavier
        gain = nn.init.calculate_gain('linear')
        nn.init.xavier_uniform_(self.input_weights, gain)
        nn.init.xavier_uniform_(self.output_projection.weight, gain)

    def project(self, inputs, index=0, chunks=1):
        ''' Produce a linear projection using the weights '''
        batch_size = inputs.shape[0]
        start = index * self.embed_dim
        end = start + chunks * self.embed_dim
        # import pdb; pdb.set_trace()
        projections = F.linear(inputs, self.input_weights[start:end]).chunk(chunks, dim=-1)

        output_projections = []
        for projection in projections:
            # transform projection to (BH x T x E)
            output_projections.append(
                projection.view(
                    batch_size,
                    -1,
                    self.num_heads,
                    self.projection_dim
                ).transpose(2, 1).contiguous().view(
                    batch_size * self.num_heads,
                    -1,
                    self.projection_dim
                )
            )

        return output_projections

    def attention(self, queries, keys, values):
        ''' Scaled dot product attention with optional masks '''
        logits = self.scale * torch.bmm(queries, keys.transpose(2, 1))

        # attended = torch.bmm(F.softmax(logits, dim=-1), values)
        attn_weights = self.sparsemax(logits)
        return attn_weights

    def forward(self, queries, keys, values):
        ''' Forward pass of the attention '''
        # pylint:disable=unbalanced-tuple-unpacking
        if same_tensor(values, keys, queries):
            values, keys, queries = self.project(values, chunks=3)
        elif same_tensor(values, keys):
            values, keys = self.project(values, chunks=2)
            queries, = self.project(queries, 2)
        else:
            values, = self.project(values, 0)
            keys, = self.project(keys, 1)
            queries, = self.project(queries, 2)

        attn_weights = self.attention(queries, keys, values)
        return attn_weights

    
def gaussian_kernel(size, sigma, device):
    """Generates a 2D Gaussian kernel."""
    coords = torch.tensor([(x - size // 2) for x in range(size)]).to(device)
    grid = coords.unsqueeze(0).repeat(size, 1)
    kernel = torch.exp(-(grid ** 2 + grid.t() ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def gaussian_blur(image, kernel_size=5, sigma=1):
    """Applies Gaussian blur to an image."""
    channels = image.shape[1]
    kernel = gaussian_kernel(kernel_size, sigma, image.device)
    # Reshape to 2d depthwise convolutional weight
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    padding = kernel_size // 2
    blurred_image = F.conv2d(image, kernel, padding=padding, groups=channels)
    return blurred_image


class MultiHeadedAttentionBlur(nn.Module):
    ''' Implement a multi-headed attention module '''
    def __init__(self, embed_dim, num_heads=1, scale=1, kernel_size=1, sigma=1):
        ''' Initialize the attention module '''
        super().__init__()

        # ensure valid inputs
        assert embed_dim % num_heads == 0, \
            f'num_heads={num_heads} should evenly divide embed_dim={embed_dim}'

        # store off the scale and input params
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.scale = scale
        self.kernel_size = kernel_size
        self.sigma = sigma
        # self.scale = self.projection_dim ** -0.5

        # Combine projections for multiple heads into a single linear layer for efficiency
        self.input_weights = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.output_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        ''' Reset parameters using xavier initialization '''
        # Initialize using Xavier
        gain = nn.init.calculate_gain('linear')
        nn.init.xavier_uniform_(self.input_weights, gain)
        nn.init.xavier_uniform_(self.output_projection.weight, gain)

    def project(self, inputs, index=0, chunks=1):
        ''' Produce a linear projection using the weights '''
        batch_size = inputs.shape[0]
        start = index * self.embed_dim
        end = start + chunks * self.embed_dim
        projections = F.linear(inputs, self.input_weights[start:end]).chunk(chunks, dim=-1)

        output_projections = []
        for projection in projections:
            # transform projection to (BH x T x E)
            output_projections.append(
                projection.view(
                    batch_size,
                    -1,
                    self.num_heads,
                    self.projection_dim
                ).transpose(2, 1).contiguous().view(
                    batch_size * self.num_heads,
                    -1,
                    self.projection_dim
                )
            )

        return output_projections

    def attention(self, queries, keys, values):
        ''' Scaled dot product attention with optional masks '''
        logits = self.scale * torch.bmm(queries, keys.transpose(2, 1))

        num_patches = int(math.sqrt(logits.shape[-1]))
        bsz, num_groups, _ = logits.shape
        logits_reshape = logits.view(bsz, num_groups, num_patches, num_patches) # this assumes the image is square
        logits_reshape_blurred = torchvision.transforms.functional.gaussian_blur(logits_reshape, self.kernel_size, self.sigma)
        attn_weights = F.softmax(logits_reshape_blurred.flatten(-2), dim=-1)
        return attn_weights

    def forward(self, queries, keys, values):
        ''' Forward pass of the attention '''
        # pylint:disable=unbalanced-tuple-unpacking
        if same_tensor(values, keys, queries):
            values, keys, queries = self.project(values, chunks=3)
        elif same_tensor(values, keys):
            values, keys = self.project(values, chunks=2)
            queries, = self.project(queries, 2)
        else:
            values, = self.project(values, 0)
            keys, = self.project(keys, 1)
            queries, = self.project(queries, 2)

        attn_weights = self.attention(queries, keys, values)
        return attn_weights


