import torch
import torch.nn as nn

from network.blocks import DoubleConv
from network.blocks import UpConvBlock


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, channels):
        super(UNet, self).__init__()

        self.channels = channels
        self.down_layers = []
        self.up_layers = []
        self.down_layers.append(DoubleConv(in_channels, channels[0]))

        for channel_idx in range(1, len(channels)):
            self.down_layers.append(
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(channels[channel_idx - 1], channels[channel_idx])
                )
            )

        for channel_idx in range(len(channels) - 1, 0, -1):
            self.up_layers.append(UpConvBlock(channels[channel_idx], channels[channel_idx - 1]))
            self.up_layers.append(DoubleConv(channels[channel_idx], channels[channel_idx - 1]))

        self.down_layers = nn.Sequential(*self.down_layers)
        self.up_layers = nn.Sequential(*self.up_layers)
        self.conv_last = nn.Conv2d(channels[0], out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        s = []
        for down_layer in self.down_layers[:-1]:
            x = down_layer(x)
            s.append(x.clone())

        x = self.down_layers[-1](x)
        s.reverse()

        for layer_idx in range(len(self.channels) - 1):
            x = self.up_layers[layer_idx * 2](x)
            x = torch.cat((x, s[layer_idx]), dim=1)
            x = self.up_layers[layer_idx * 2 + 1](x)

        x = self.conv_last(x)

        return x
