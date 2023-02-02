# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

from models.attentions import Attention2d
from models.layers import conv1x1


class LocalAttention(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 window_size=7, k=1,
                 heads=8, dim_head=32, dropout=0.0):
        super().__init__()
        self.attn = Attention2d(dim_in, dim_out,
                                heads=heads, dim_head=dim_head, dropout=dropout, k=k)
        self.window_size = window_size

        self.rel_index = (self.rel_distance(window_size) + window_size - 1).long()
        self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1) * 0.02)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        p = self.window_size
        n1 = h // p
        n2 = w // p

        mask = torch.zeros(p ** 2, p ** 2, device=x.device) if mask is None else mask
        mask = mask + self.pos_embedding[self.rel_index[:, :, 0], self.rel_index[:, :, 1]]

        x = rearrange(x, "b c (n1 p1) (n2 p2) -> (b n1 n2) c p1 p2", p1=p, p2=p)
        x, attn = self.attn(x, mask)
        x = rearrange(x, "(b n1 n2) c p1 p2 -> b c (n1 p1) (n2 p2)", n1=n1, n2=n2, p1=p, p2=p)

        return x, attn

    @staticmethod
    def rel_distance(window_size):
        i = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
        d = i[None, :, :] - i[:, None, :]

        return d


class AttentionBlockB(nn.Module):
    # Attention block with pre-activation.
    # We use this block by default.
    expansion = 1

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=64, dropout=0.0, sd=0.0,
                 stride=1, window_size=7, k=1, norm=nn.BatchNorm2d, activation=nn.GELU,
                 **block_kwargs):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out
        attn = partial(LocalAttention, window_size=window_size, k=k)
        width = dim_in // self.expansion

        self.shortcut = []
        if stride != 1 or dim_in != dim_out * self.expansion:
            self.shortcut.append(conv1x1(dim_in, dim_out * self.expansion, stride=stride))
        self.shortcut = nn.Sequential(*self.shortcut)
        self.norm1 = norm(dim_in)
        self.relu = activation()

        self.conv = nn.Conv2d(dim_in, width, kernel_size=1, bias=False)
        self.norm2 = norm(width)
        self.attn = attn(width, dim_out * self.expansion, heads=heads, dim_head=dim_head, dropout=dropout)
        self.sd = DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x):
        if len(self.shortcut) > 0:
            x = self.norm1(x)
            x = self.relu(x)
            skip = self.shortcut(x)
        else:
            skip = self.shortcut(x)
            x = self.norm1(x)
            x = self.relu(x)

        x = self.conv(x)
        x = self.norm2(x)
        x, attn = self.attn(x)

        x = self.sd(x) + skip

        return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class CCTBlock(nn.Module):

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6,
                 heads=8, dim_head=64, dropout=0.0, sd=0.0,
                 stride=1, window_size=7, k=1, norm=nn.LayerNorm,
                 activation=nn.GELU, scale_factor=0.8):
        super().__init__()
        self.conv_block = Block(dim=dim, drop_path=drop_path, layer_scale_init_value=layer_scale_init_value)
        attn_norm = norm
        self.attention_block = AttentionBlockB(dim_in=dim, dim_out=dim,heads=heads, dim_head=dim_head,
                                               stride=stride, dropout=dropout, sd=drop_path, window_size=window_size,
                                               k=k,
                                               #FIXME: Make Layer Norm work
                                               #norm=attn_norm,
                                               activation=activation)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv_block(x)
        x = self.attention_block(x)
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor)
        return x


class CCTNet(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], heads=[3, 6, 12, 24],
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 stem=4, **kwargs
                 ):
        super().__init__()
        self.default_cfg = {}

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=stem, stride=stem),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
        #        LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
        #        nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                nn.Identity()
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[CCTBlock(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value, heads=heads[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


@register_model
def cctnet_tiny(pretrained=False, in_22k=False, **kwargs):
    model = CCTNet(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def cctnet_small(pretrained=False, in_22k=False, **kwargs):
    model = CCTNet(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def cctnet_base(pretrained=False, in_22k=False, **kwargs):
    model = CCTNet(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def cctnet_large(pretrained=False, in_22k=False, **kwargs):
    model = CCTNet(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def cctnet_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = CCTNet(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model
