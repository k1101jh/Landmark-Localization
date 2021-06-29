import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttentionUNet, self).__init__()

        self.ln = [64, 128, 256, 512]

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(in_channels=img_ch, out_channels=self.ln[0])
        self.conv2 = ConvBlock(in_channels=self.ln[0], out_channels=self.ln[1])
        self.conv3 = ConvBlock(in_channels=self.ln[1], out_channels=self.ln[2])
        self.conv4 = ConvBlock(in_channels=self.ln[2], out_channels=self.ln[3])

        self.up_conv3_1 = UpConvBlock(in_channels=self.ln[3], out_channels=self.ln[2])
        self.att3 = AttentionBlock(f_g=self.ln[2], f_l=self.ln[2], f_int=self.ln[1])
        self.up_conv3_2 = ConvBlock(in_channels=self.ln[3], out_channels=self.ln[2])

        self.up_conv2_1 = UpConvBlock(in_channels=self.ln[2], out_channels=self.ln[1])
        self.att2 = AttentionBlock(f_g=self.ln[1], f_l=self.ln[1], f_int=self.ln[0])
        self.up_conv2_2 = ConvBlock(in_channels=self.ln[2], out_channels=self.ln[1])

        self.up_conv1_1 = UpConvBlock(in_channels=self.ln[1], out_channels=self.ln[0])
        self.att1 = AttentionBlock(f_g=self.ln[0], f_l=self.ln[0], f_int=self.ln[0] // 2)
        self.up_conv1_2 = ConvBlock(in_channels=self.ln[1], out_channels=self.ln[0])

        self.conv_last = nn.Conv2d(self.ln[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        conv1 = self.conv1(x)

        x = self.maxpool(conv1)
        conv2 = self.conv2(x)

        x = self.maxpool(conv2)
        conv3 = self.conv3(x)

        x = self.maxpool(conv3)
        conv4 = self.conv4(x)

        d = self.up_conv3_1(conv4)
        x = self.att3(g=d, x=conv3)
        d = torch.cat((x, d), dim=1)
        d = self.up_conv3_2(d)

        d = self.up_conv2_1(d)
        x = self.att2(g=d, x=conv2)
        d = torch.cat((x, d), dim=1)
        d = self.up_conv2_2(d)

        d = self.up_conv1_1(d)
        x = self.att1(g=d, x=conv1)
        d = torch.cat((x, d), dim=1)
        d = self.up_conv1_2(d)

        d = self.conv_last(d)

        return d
