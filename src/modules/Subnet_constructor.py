import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.module_util as mutil

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, use_linear, init='xavier', gc=32, bias=True, residual= False):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.residual = residual

        self.use_linear = use_linear
        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        if not self.use_linear:
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(torch.cat((x, x1), 1))
            x3 = self.conv3(torch.cat((x, x1, x2), 1))
            x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        if not self.residual:
            return x5
        else:
            return x+x5

class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out, use_linear, residual= False):
        super(ResBlock, self).__init__()
        feature = 64
        self.conv1 = nn.Conv2d(channel_in, feature, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(feature, feature, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d((feature+channel_in), channel_out, kernel_size=3, padding=1)
        self.residual = residual
        self.use_linear = use_linear

    def forward(self, x):
        if not self.use_linear:
            residual = self.relu1(self.conv1(x))
            residual = self.relu1(self.conv2(residual))
        else:
            residual = self.conv1(x)
            residual = self.conv2(residual)
        input = torch.cat((x, residual), dim=1)
        out = self.conv3(input)
        if not self.residual:
            return out
        else:
            return x+out

def subnet(net_structure, init='xavier',use_linear=False, residual=False, mode='Dense'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                if mode=='Dense':
                    return DenseBlock(channel_in, channel_out, use_linear, init, residual=residual)
                else:
                    return ResBlock(channel_in, channel_out, use_linear, residual=residual)

            else:
                if mode=='Dense':
                    return DenseBlock(channel_in, channel_out, use_linear, residual=residual)
                else:
                    return ResBlock(channel_in, channel_out, use_linear, residual=residual)
        else:
            return None

    return constructor
