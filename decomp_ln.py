import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import LayerNorm2d


class Decomp_LN(nn.Module):
    def __init__(self, normalized_shape, channels_last=False, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.channels_last = channels_last
        self.eps = eps

        # Learnable affine parameters (gamma and beta)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # Handle channels_last (NHWC format for non-2D data)
        if self.channels_last:
            # Expect shape: (batch, ..., channels)
            # Normalize along the last dimension (channels)
            norm_axis = -1
        else:
            # Expect shape: (batch, channels, ...)
            # Normalize along the channel dimension (dim=1)
            norm_axis = 1

        # Compute scale per feature vector (across channels)
        scale = torch.norm(x, p=2, dim=norm_axis, keepdim=True)

        # Unit direction vector
        direction = x / (scale + self.eps)

        # Normalize direction across channels (last dim)
        mean = direction.mean(dim=norm_axis, keepdim=True)
        std = direction.std(dim=norm_axis, keepdim=True)
        normed_direction = (direction - mean) / (std + self.eps)

        # Reapply scale
        rescaled = normed_direction * F.sigmoid(scale) * 2

        if self.channels_last:
            # Affine transformation (broadcasts correctly over last dim)
            out = rescaled * self.weight + self.bias

        else:
            # Affine transformation (broadcasts over channels)
            out = rescaled * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

        return out


def convert_ln_to_decomp_ln(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        # channels_last = not isinstance(module, LayerNorm2d)
        module_output = Decomp_LN(
            module.normalized_shape, channels_last=not isinstance(module, LayerNorm2d)
        )
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_decomp_ln(child))
    del module
    return module_output
