import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        # nn.ReLU(inplace=True),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.LeakyReLU(0.2, inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.LeakyReLU(0.2, inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    
def initialize_weights(net):
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

# class Discriminator(nn.Module):
#     def __init__(self,  args):
#         super(Discriminator, self).__init__()
#         # self.name = f'discriminator_{args.dataset}'
#         self.bias = False
#         channels = 32

#         layers = [
#             nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=self.bias),
#             nn.LeakyReLU(0.2, True)
#         ]

#         for i in range(3):
#             layers += [
#                 nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=self.bias),
#                 nn.LeakyReLU(0.2, True),
#                 nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=1, padding=1, bias=self.bias),
#                 nn.InstanceNorm2d(channels * 4),
#                 nn.LeakyReLU(0.2, True),
#             ]
#             channels *= 4

#         layers += [
#             nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=self.bias),
#             nn.InstanceNorm2d(channels),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=self.bias),
#         ]

#         if True:
#             for i in range(len(layers)):
#                 if isinstance(layers[i], nn.Conv2d):
#                     layers[i] = spectral_norm(layers[i])

#         self.discriminate = nn.Sequential(*layers)

#         initialize_weights(self)

#     def forward(self, img):
#         return self.discriminate(img)
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)