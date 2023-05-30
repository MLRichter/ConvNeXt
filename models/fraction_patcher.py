import torch
from torch import nn


class FracConv2(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_up: int, kernel_down: int, stride_up: int, down_stride: int):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_up = kernel_up
        self.kernel_down = kernel_down
        self.stride_up = stride_up
        self.down_stride = down_stride
        self.tansposed_conv = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=self.kernel_up,
            stride=self.stride_up
        )
        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_down,
            stride=self.down_stride
        )
        self.act1 = nn.GELU()
        self.act2 = nn.GELU()

        # output_size = (input_size - 1) * stride - 2*padding - (kernel_size - 1) + 1

    def forward(self, x):
        upsampled = self.act1(self.tansposed_conv(x))
        downsampled = self.act2(self.conv(upsampled))
        #shape = downsampled.size()[-1], downsampled.size(-2)
        #interpolated = nn.functional.interpolate(x, shape)
        return downsampled #+ interpolated


class FracConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, alpha: float):
        super().__init__()
        self.alpha = alpha
        self.beta = 2 if alpha >= 0.5 else 4
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.beta, stride=self.beta)

    @property
    def upsample_alpha(self):
        return (self.alpha) * self.beta

    def forward(self, x):
        upsampled = nn.functional.interpolate(x, scale_factor=self.upsample_alpha, mode="bilinear")
        downsampled = self.conv(upsampled)
        return downsampled


if __name__ == '__main__':
    fc = FracConv2(in_channels=32, out_channels=32, kernel_up=3, kernel_down=3, stride_up=1, down_stride=3)
    x = torch.ones((1, 32, 100, 100))

    #fc = FracConv(32, 32, 0.8)
    #x = torch.zeros((1, 32, 100, 100))
    x_out = fc(x)
    print(x_out.size())