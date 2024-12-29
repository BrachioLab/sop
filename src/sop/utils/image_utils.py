import torch

# mask with random indices
def occlude_input_with_random_pixels(x, m):
    bsz, C, H, W = x.size()

    # Generate random indices for the height and width dimensions
    random_h = torch.randint(0, H, (bsz,))
    random_w = torch.randint(0, W, (bsz,))

    # Create batch indices
    batch_indices = torch.arange(bsz)
    
    # Gather the values, resulting in a tensor of shape (bsz, C)
    selected_values = x[batch_indices, :, random_h, random_w]
    masked_values = (1 - m[:, None]) * selected_values[:, :, None, None]
    indices_occlude = (1 - m[:, None]).expand(bsz, C, H, W) == 1
    x[indices_occlude] = masked_values[indices_occlude]
    return x