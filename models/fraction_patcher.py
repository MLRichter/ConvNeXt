import torch
from torch import nn


class FracConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, alpha: float):
        super().__init__()
        self.alpha = alpha
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)

    @property
    def upsample_alpha(self):
        return (self.alpha) * 2

    def forward(self, x):
        upsampled = nn.functional.interpolate(x, scale_factor=self.upsample_alpha, mode="bilinear")
        downsampled = self.conv(upsampled)
        return downsampled


if __name__ == '__main__':
    fc = FracConv(32, 32, 0.8)
    x = torch.zeros((1, 32, 100, 100))
    x_out = fc(x)
    print(x_out.size())