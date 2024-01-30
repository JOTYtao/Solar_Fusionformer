import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
import math
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import requests
from io import BytesIO

def get_gaussian_filter(kernel_size=3, sigma=1, channels=3, feature_map=None):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2. * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    if feature_map is not None:
        local_density = feature_map.unfold(0, kernel_size, 1).unfold(1, kernel_size, 1)
        local_density = local_density.mean(dim=[2, 3], keepdim=True)
        local_density_normalized = (local_density - local_density.min()) / (local_density.max() - local_density.min())
        gaussian_kernel = gaussian_kernel * local_density_normalized
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    padding = (kernel_size - 1) // 2
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=padding)
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter
class FeatureMap:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []
        self.feature_maps = []
        self.hooks = []
        self.register_hooks()

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def forward_hook(self, module, input, output):
        self.feature_maps.append(output)
        output.register_hook(self.save_gradient)

    def register_hooks(self):
        for layer_name in self.target_layers:
            layer = dict([*self.model.named_modules()])[layer_name]
            self.hooks.append(layer.register_forward_hook(self.forward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __call__(self, x):
        self.feature_maps = []
        self.gradients = []
        with torch.no_grad():
            _ = self.model(x)
        return self.feature_maps


class ResNetFeatureExtractor:
    def __init__(self, target_layers):
        self.model = models.resnet18(pretrained=True)
        self.model.eval()
        self.feature_extractor = FeatureMap(self.model, target_layers)
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    def extract_features(self, image):
        input_image = self.download_image(image)
        input_tensor = self.preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        feature_maps = self.feature_extractor(input_batch)
        self.feature_extractor.remove_hooks()

        return feature_maps