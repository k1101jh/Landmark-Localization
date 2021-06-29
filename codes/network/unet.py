import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, n_input, n_class):
        super().__init__()

        self.ln = [64, 128, 256, 512]

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_down1 = self.double_conv(n_input, self.ln[0])
        self.dconv_down2 = self.double_conv(self.ln[0], self.ln[1])
        self.dconv_down3 = self.double_conv(self.ln[1], self.ln[2])
        self.dconv_down4 = self.double_conv(self.ln[2], self.ln[3])

        self.dconv_up3 = self.double_conv(self.ln[2] + self.ln[3], self.ln[2])
        self.dconv_up2 = self.double_conv(self.ln[1] + self.ln[2], self.ln[1])
        self.dconv_up1 = self.double_conv(self.ln[1] + self.ln[0], self.ln[0])

        self.conv_last = nn.Conv2d(self.ln[0], n_class, 1)

    @staticmethod
    def double_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        conv1 = self.dconv_down1(x)

        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)

        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)

        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)

        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)
        out = self.conv_last(x)

        return out
