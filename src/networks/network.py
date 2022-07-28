import torch
import torch.nn as nn
from torch.nn import init
from utils.util import swap_axis
from networks.Zupsamplers import common


class Generator(nn.Module):
    def __init__(self, conf):
        super(Generator, self).__init__()
        struct = conf.G_structure
        # First layer - Converting RGB image to latent space
        self.first_layer = nn.Conv2d(in_channels=1, out_channels=conf.G_chan, kernel_size=struct[0], bias=False)

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            feature_block += [nn.Conv2d(in_channels=conf.G_chan, out_channels=conf.G_chan, kernel_size=struct[layer], bias=False)]
        self.feature_block = nn.Sequential(*feature_block)
        # Final layer - Down-sampling and converting back to image
        self.final_layer = nn.Conv2d(in_channels=conf.G_chan, out_channels=1, kernel_size=struct[-1],
                                     stride=int(1 / conf.scale_factor), bias=False)

        # Calculate number of pixels shaved in the forward pass
        self.output_size = self.forward(torch.FloatTensor(torch.ones([1, 1, conf.input_crop_size, conf.input_crop_size]))).shape[-1]
        self.forward_shave = int(conf.input_crop_size * conf.scale_factor) - self.output_size

    def forward(self, input_tensor):
        # Swap axis of RGB image for the network to get a "batch" of size = 3 rather the 3 channels
        input_tensor = swap_axis(input_tensor)
        downscaled = self.first_layer(input_tensor)
        features = self.feature_block(downscaled)
        output = self.final_layer(features)
        return swap_axis(output)

class DualDownsampler(nn.Module):
    def __init__(self, conf):
        super(DualDownsampler, self).__init__()
        struct = conf.G_structure
        # First layer - Converting RGB image to latent space
        self.first_layer = nn.Conv2d(in_channels=1, out_channels=conf.G_chan, kernel_size=struct[0], padding=struct[0]// 2, bias=False)

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            feature_block += [nn.Conv2d(in_channels=conf.G_chan, out_channels=conf.G_chan, kernel_size=struct[layer], padding=struct[layer]// 2, bias=False)]
        self.feature_block = nn.Sequential(*feature_block)
        # Final layer - Down-sampling and converting back to image
        self.final_layer = nn.Conv2d(in_channels=conf.G_chan, out_channels=1, kernel_size=struct[-1], padding=struct[-1]// 2,
                                     stride=int(1 / conf.scale_factor), bias=False)

        # Calculate number of pixels shaved in the forward pass
        self.output_size = self.forward(torch.FloatTensor(torch.ones([1, 1, conf.input_crop_size, conf.input_crop_size]))).shape[-1]
        self.forward_shave = int(conf.input_crop_size * conf.scale_factor) - self.output_size

    def forward(self, input_tensor):
        # Swap axis of RGB image for the network to get a "batch" of size = 3 rather the 3 channels
        input_tensor = swap_axis(input_tensor)
        downscaled = self.first_layer(input_tensor)
        features = self.feature_block(downscaled)
        output = self.final_layer(features)
        return swap_axis(output)

class ZSSRUpsampler(nn.Module):
    def __init__(self, conf, input_channels=3, kernel_size=3, channels=64):
        super(ZSSRUpsampler, self).__init__()
        self.conf = conf
        self.upsampler = nn.Upsample(scale_factor=1/self.conf.scale_factor, mode='bicubic', align_corners=True)
        self.conv0 = nn.Conv2d(input_channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv6 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.conv7 = nn.Conv2d(channels, input_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)

        self.relu = nn.ReLU()

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(1, b * c, h, w)

        x = self.upsampler(x)
        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.conv7(x)

        _, _, h, w = x.shape
        x = x.reshape(b, c, h, w)
        return x


class ImageDiscriminator(nn.Module):
    def __init__(self, conf):
        super(ImageDiscriminator, self).__init__()
        self.channel = 3

        k_sizes = [5,5,5,5]
        ndf = 64
        in_nc = [self.channel, ndf, ndf*2]
        out_nc = [ndf, ndf*2, ndf*4]
        conv_block = []
        for k_size, in_c, out_c in zip(k_sizes, in_nc, out_nc):
            conv_block += [nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=k_size, stride=2, padding=2, bias=False),
                              nn.BatchNorm2d(out_c),
                              nn.LeakyReLU(0.2, inplace=True)]
        self.conv_block = nn.Sequential(*conv_block)
        self.final_layer = nn.Sequential(
            nn.Conv2d(in_channels=ndf*4, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        features = self.conv_block(input)
        output = self.final_layer(features)
        return output.view(-1)



class Discriminator(nn.Module):
    def __init__(self, conf):
        super(Discriminator, self).__init__()

        self.channel = 3
        # First layer - Convolution (with no ReLU)
        k_size = conf.D_kernel_size

        k_lst = [1]*(conf.D_n_layers-1)
        current_id = 0
        while k_size > 1:
            k_lst[current_id] = k_size if k_size < 7 else 7
            current_id += 1
            k_size -= 6

        print(k_lst)

        self.first_layer = nn.utils.spectral_norm(nn.Conv2d(in_channels=self.channel, out_channels=conf.D_chan, kernel_size=k_lst[0], bias=True))
        feature_block = []  # Stacking layers with 1x1 kernels
        for i in range(1, conf.D_n_layers - 1):
            feature_block += [nn.utils.spectral_norm(nn.Conv2d(in_channels=conf.D_chan, out_channels=conf.D_chan, kernel_size=k_lst[i], bias=True)),
                              nn.BatchNorm2d(conf.D_chan),
                              nn.ReLU(True)]
        self.feature_block = nn.Sequential(*feature_block)
        # if not conf.mixup:
        #     self.final_layer = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(in_channels=conf.D_chan, out_channels=1, kernel_size=1, bias=True)),
        #                                      nn.Sigmoid())
        # else:
        #     self.final_layer = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(in_channels=conf.D_chan, out_channels=2, kernel_size=1, bias=True)),
        #                                      nn.Softmax())
        self.final_layer = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels=conf.D_chan, out_channels=1, kernel_size=1, bias=True)),
            nn.Sigmoid())

        # Calculate number of pixels shaved in the forward pass
        self.forward_shave = conf.input_crop_size - self.forward(torch.FloatTensor(torch.ones([1, self.channel, conf.input_crop_size, conf.input_crop_size]))).shape[-1]

    def forward(self, input_tensor, return_features=False):
        receptive_extraction = self.first_layer(input_tensor)
        features = self.feature_block(receptive_extraction)
        if not return_features:
            return self.final_layer(features)
        else:
            return self.final_layer(features), features


class Upsample(nn.Module):
    def __init__(self, conf):
        super(Upsample, self).__init__()

        struct = conf.G_structure
        self.scale = 1/conf.scale_factor
        self.n_colors = int(self.scale ** 2 - 1)


        self.first_layer = nn.Conv2d(in_channels=self.n_colors*3, out_channels=conf.G_chan, kernel_size=struct[0], bias=False, padding=struct[0]//2)

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct) - 1):
            feature_block += [nn.Conv2d(in_channels=conf.G_chan, out_channels=conf.G_chan, kernel_size=struct[layer], bias=False, padding=struct[layer]//2)]
        self.feature_block = nn.Sequential(*feature_block)
        # Final layer - Down-sampling and converting back to image
        self.final_layer = nn.Conv2d(in_channels=conf.G_chan, out_channels=int(self.n_colors*3), kernel_size=struct[-1], bias=False, padding=struct[-1]//2)

        # self.output_size = self.forward(torch.FloatTensor(torch.ones([3, 1, conf.input_crop_size, conf.input_crop_size]))).shape[-1]
        # self.crop_size = torch.FloatTensor(torch.ones([1, 1, conf.input_crop_size, conf.input_crop_size])).shape[-1]
        # self.forward_shave = int(conf.input_crop_size * self.scale) - self.output_size

    def forward(self, input_tensor):
        input_tensor = self.swap_and_norm(input_tensor)
        input_tensor = nn.Upsample(scale_factor=self.scale, mode='nearest')(input_tensor)
        downscaled = self.first_layer(input_tensor)
        features = self.feature_block(downscaled)
        output = self.final_layer(features) + input_tensor
        return self.swap_and_norm(output, 'denorm')

    def swap_and_norm(self, x, mode='norm'):
        if mode == 'norm':
            # TODO
            # x = swap_axis(x)
            b, c, h, w = x.shape
            x = x.reshape(1, b*c, h, w)

            self.mean = torch.mean(x, (2, 3))
            self.std = torch.std(x, (2, 3))
            x = common.norm(x, self.mean, self.std)
        else:
            x = common.norm(x, self.mean, self.std, 'denorm')
            # TODO
            b, c, h, w = x.shape
            # x = swap_axis(x)
            x = x.reshape(self.n_colors, int(c//self.n_colors), h, w)
        return x


class UpsampleDiscriminator(nn.Module):
    def __init__(self, in_c, conf):
        super(UpsampleDiscriminator, self).__init__()

        # First layer - Convolution (with no ReLU)
        self.first_layer = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_c, out_channels=conf.D_chan, kernel_size=conf.D_kernel_size, bias=True))
        feature_block = []  # Stacking layers with 1x1 kernels
        for _ in range(1, conf.D_n_layers - 1):
            feature_block += [nn.utils.spectral_norm(nn.Conv2d(in_channels=conf.D_chan, out_channels=conf.D_chan, kernel_size=1, bias=True)),
                              nn.BatchNorm2d(conf.D_chan),
                              nn.ReLU(True)]
        self.feature_block = nn.Sequential(*feature_block)
        # if not conf.mixup:
        #     self.final_layer = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(in_channels=conf.D_chan, out_channels=1, kernel_size=1, bias=True)),
        #                                      nn.Sigmoid())
        # else:
        #     self.final_layer = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(in_channels=conf.D_chan, out_channels=2, kernel_size=1, bias=True)),
        #                                      nn.Softmax())
        self.final_layer = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels=conf.D_chan, out_channels=2, kernel_size=1, bias=True)),
            nn.Softmax())

        # Calculate number of pixels shaved in the forward pass
        self.forward_shave = conf.input_crop_size - self.forward(torch.FloatTensor(torch.ones([1, in_c, conf.input_crop_size, conf.input_crop_size]))).shape[-1]

    def forward(self, input_tensor):
        receptive_extraction = self.first_layer(input_tensor)
        features = self.feature_block(receptive_extraction)
        return self.final_layer(features)

def weights_init_D(m):
    """ initialize weights of the discriminator """
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 0.1)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif class_name.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weights_init_G(m):
    """ initialize weights of the generator """
    if m.__class__.__name__.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 0.1)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
