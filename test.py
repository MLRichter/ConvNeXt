import numpy as np
import torch
from torch.autograd import gradcheck
import torch
from torch import nn
import torch.nn.functional as F

def normalize_floats(float_list, target_value):
    scaling_factor = 1 / np.power(float_list.prod() * (1/target_value), (1/len(float_list)))
    normalized_floats = (float_list * scaling_factor)
    return normalized_floats

def normalize_capped_float(float_list: np.ndarray, target_value):
    normed_float = normalize_floats(float_list, target_value)
    normed_float = np.asarray([min(max(x, 0.5), 1.0) for x in normed_float])
    return normalize_floats(normed_float, target_value)


if False:

    floats = np.asarray([0.6, 0.8, 0.6, 0.8, 0.6])
    target_value = 7/224
    normalized_floats = normalize_capped_float(floats, target_value)
    print(normalized_floats, normalized_floats.prod(), floats.prod(), 7/224)



def linear_interpolation(image, new_size):
    # Get the dimensions of the original image
    height, width = image.shape[-2:]

    # Calculate the scale factors for resizing
    scale_height = new_size[0] / height
    scale_width = new_size[1] / width

    # Create a grid of normalized coordinates for the new image
    new_height, new_width = new_size
    grid_y = torch.linspace(0, 1, new_height).view(1, new_height, 1)
    grid_x = torch.linspace(0, 1, new_width).view(1, 1, new_width)

    # Generate the sampling grid
    grid = torch.cat((grid_x.expand(1, new_height, new_width),
                      grid_y.expand(1, new_height, new_width)), dim=0)

    # Reshape the image for interpolation
    image = image.view(1, 1, height, width)

    # Generate the interpolated image
    interpolated_image = F.grid_sample(image, grid, align_corners=True)

    return interpolated_image.squeeze()


class ResizeModule(nn.Module):

    def __init__(self, alpha):
        super(ResizeModule, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.alpha_weight = 0.0001

    def forward(self, x):
        shape = x.shape
        height = self.alpha * x.shape[0]
        width = self.alpha * x.shape[1]
        # resize the tensor using F.interpolate and return
        return linear_interpolation(x, (height, width))


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableNearestInterpolation(nn.Module):
    def __init__(self, alpha):
        super(DifferentiableNearestInterpolation, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, img):
        B, C, H, W = img.shape
        new_H = self.alpha * H
        new_W = self.alpha * W

        # create meshgrid in target image
        grid_y, grid_x = torch.meshgrid(torch.arange(new_H), torch.arange(new_W))

        # scale grid coordinates in source image
        grid_y = grid_y.float() / self.alpha
        grid_x = grid_x.float() / self.alpha

        # compute nearest neighbors
        grid_y_nearest = torch.round(grid_y)
        grid_x_nearest = torch.round(grid_x)

        # clamp values to be within image bounds
        grid_y_clamped = torch.clamp(grid_y_nearest, 0, H-1)
        grid_x_clamped = torch.clamp(grid_x_nearest, 0, W-1)

        # gather values from source image
        new_img = img[:, :, grid_y_clamped.long(), grid_x_clamped.long()]

        return new_img





# Instantiate ResizeModule
resize_module = DifferentiableNearestInterpolation(alpha=.4)
resize_module = resize_module.double()  # gradcheck needs double precision

# Randomly initialize a tensor for input
input = torch.randn(2, 7, 10, 10, dtype=torch.double, requires_grad=True)

# Use gradcheck to verify the gradients
gradcheck_passed = gradcheck(resize_module, (input,), eps=1e-6, atol=1e-4)
print(f"Gradcheck passed: {gradcheck_passed}")

# Run a forward and backward pass
output = resize_module(input)
print("Input shape:", input.shape)
print("Output shape:", output.shape)

output.sum().backward()  # Sum needed to create a scalar for backward()
print("Gradient for alpha:", resize_module.alpha.grad)

