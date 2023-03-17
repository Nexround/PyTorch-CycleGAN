
import torch.nn as nn
from model.conv_blocks import DownConv
from model.conv_blocks import UpConv
from model.conv_blocks import SeparableConv2D
from model.conv_blocks import InvertedResBlock
from model.conv_blocks import ConvBlock


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        bias = False

        self.encode_blocks = nn.Sequential(
            ConvBlock(3, 64, bias=bias),
            ConvBlock(64, 128, bias=bias),
            DownConv(128, bias=bias),
            ConvBlock(128, 128, bias=bias),
            SeparableConv2D(128, 256, bias=bias),
            DownConv(256, bias=bias),
            ConvBlock(256, 256, bias=bias),
        )

        self.res_blocks = nn.Sequential(
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
        )

        self.decode_blocks = nn.Sequential(
            ConvBlock(256, 128, bias=bias),
            UpConv(128, bias=bias),
            SeparableConv2D(128, 128, bias=bias),
            ConvBlock(128, 128, bias=bias),
            UpConv(128, bias=bias),
            ConvBlock(128, 64, bias=bias),
            ConvBlock(64, 64, bias=bias),
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.Tanh(),
        )

        self.initialize_weights(self) # 网络初始化

    def initialize_weights(self,net):
        for m in net.modules():
            try:
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0, 0.02)
                    m.bias.data.zero_()
                elif isinstance(m, nn.ConvTranspose2d):
                    m.weight.data.normal_(0, 0.02)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.02)
                    m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            except Exception as e:
                # print(f'SKip layer {m}, {e}')
                pass

    def forward(self, x):
        out = self.encode_blocks(x)
        out = self.res_blocks(out)
        img = self.decode_blocks(out)

        return img

