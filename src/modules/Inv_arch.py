import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.util import swap_axis
from modules.inv_conv import InvertibleConv1x1


class SpaceToDepth(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x, rev=False):
        N, C, H, W = x.size()
        if not rev:
            x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
            x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
            x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        else:
            x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
            x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
            x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1 / 16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.channel_in)

    def jacobian(self, x, rev=False):
        return self.last_jac


class InvBlockTriChannel(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_lr, channel_split_res, couple_layer, clamp=1.):
        super(InvBlockTriChannel, self).__init__()
        #####################################################
        # Channel for x1(LR image) : channel_split_lr (1)   #
        # Channel for x2(Z feats)  : channel_num-res-lr (2) #
        # Channel for x3(Res image): channel_split_res (1)  #
        #####################################################
        self.split_lr = channel_split_lr
        self.split_res = channel_split_res
        self.split_z = channel_num - channel_split_res - channel_split_lr

        self.couple_layer = couple_layer
        self.clamp = clamp

        self.F = subnet_constructor(self.split_z, self.split_lr)
        self.G = subnet_constructor(self.split_lr, self.split_z)
        self.H = subnet_constructor(self.split_lr, self.split_z)

        # R for residual path module
        self.R = subnet_constructor(self.split_res, self.split_lr)


    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_lr), x.narrow(1, self.split_lr, self.split_z))
        x3 = x.narrow(1, self.split_lr+self.split_z, self.split_res)

        if self.couple_layer=='affine':
            if not rev:
                y1 = x1 + self.F(x2)
                self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
                y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
                # Residual Part
                y3 = x3
                y1 = y1 + self.R(x3)
            else:
                self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
                y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
                # Residual Part
                y3 = x3
                y1 = x1 - self.F(y2) -self.R(y3)
        elif self.couple_layer=='additive':
            if not rev:
                y1 = x1 + self.F(x2)
                y2 = x2 + self.G(y1)
                # Residual Part
                y3 = x3
                y1 = y1 + self.R(x3)
            else:
                y2 = x2 - self.G(x1)
                # Residual Part
                y3 = x3
                y1 = x1 - self.F(y2) - self.R(y3)


        elif self.couple_layer == 'none':
            y1 = x1
            y2 = x2
            y3 = x3

        return torch.cat((y1, y2, y3), 1)


class InvBlockExp(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, couple_layer, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num                 # 1
        self.split_len2 = channel_num - channel_split_num   # 3

        self.couple_layer = couple_layer

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        if self.couple_layer=='affine':
            if not rev:
                y1 = x1 + self.F(x2)
                self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
                y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
            else:
                self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
                y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
                y1 = x1 - self.F(y2)
        elif self.couple_layer=='additive':
            if not rev:
                y1 = x1 + self.F(x2)
                y2 = x2 + self.G(y1)
            else:
                y2 = x2 - self.G(x1)
                y1 = x1 - self.F(y2)
        elif self.couple_layer == 'none':
            y1 = x1
            y2 = x2

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]


class InvRescaleNet(nn.Module):
    def __init__(self, channel_in=1, channel_out=1, subnet_constructor=None, block_num=[], down_num=2, couple_layer='affine',\
                 down_sample=['Haar'], tri_channel=False, use_res=False, down_mode='multiple_model', visualize=False):
        super(InvRescaleNet, self).__init__()

        operations = []

        if down_mode == 'multiple_model':
            current_channel = channel_in
            for i in range(down_num):
                if down_sample[i] == 'Haar':
                    operations.append(HaarDownsampling(current_channel))
                elif down_sample[i] == 'space2depth':
                    operations.append(SpaceToDepth(2))
                elif down_sample[i] == 'Glow':
                    operations.append(SpaceToDepth(2))
                    operations.append(InvertibleConv1x1(current_channel*4, LU_decomposed=True))

                current_channel *= 4
                for j in range(block_num[i]):
                    if tri_channel:
                        b = InvBlockTriChannel(subnet_constructor, current_channel, channel_out, channel_out, couple_layer)
                    else:
                        b = InvBlockExp(subnet_constructor, current_channel, channel_out, couple_layer)
                    operations.append(b)

        elif down_mode == 'single_model':
            current_channel = channel_in
            for i in range(down_num):
                if down_sample[i] == 'Haar':
                    b = HaarDownsampling(current_channel)
                elif down_sample[i] == 'space2depth':
                    b = SpaceToDepth(2)
                operations.append(b)
                current_channel *= 4
            for j in range(block_num[i]):
                if tri_channel:
                    b = InvBlockTriChannel(subnet_constructor, current_channel, channel_out, channel_out, couple_layer)
                else:
                    b = InvBlockExp(subnet_constructor, current_channel, channel_out, couple_layer)
                operations.append(b)

        self.operations = nn.ModuleList(operations)

        self.use_res = use_res
        self.downsampler = nn.Upsample(scale_factor=1/2**down_num, mode='bicubic', align_corners=True)
        self.upsampler = nn.Upsample(scale_factor=2**down_num, mode='bicubic', align_corners=True)

    def forward(self, x, rev=False, print_res=None):
        if not self.use_res:
            out = swap_axis(x)
            if not rev:
                for op in self.operations:
                    out = op.forward(out, rev)

            else:
                for op in reversed(self.operations):
                    out = op.forward(out, rev)
            return swap_axis(out)

        else:
            if not rev:
                x_llr = self.downsampler(x)
                x_lr = self.upsampler(x_llr)
                x_res = x - x_lr
                out = swap_axis(x_res)
                for op in self.operations:
                    out = op.forward(out, rev)

                out = swap_axis(out)

                if print_res != None:
                    from utils.util import write_image
                    write_image(x_res[:1, :, :, :], print_res+'/res_lr.png')
                    write_image(out[:1, :, :, :], print_res+'/res_llr.png')
                    write_image(x_lr[:1, :, :, :], print_res+'/bic_lr.png')
                    write_image(x_llr[:1, :, :, :], print_res+'/bic_llr.png')

                out[0] += x_llr[0]
                return out
            else:
                x_llr = self.downsampler(x)
                x_lr = self.upsampler(x_llr)
                x_hr = self.upsampler(x)
                x[0] -= x_lr[0]
                out = swap_axis(x)
                for op in reversed(self.operations):
                    out = op.forward(out, rev)
                out = swap_axis(out)

                if print_res != None:
                    from utils.util import write_image
                    write_image(x[:1, :, :, :], print_res+'/res_lr.png')
                    write_image(out[:1, :, :, :], print_res+'/res_hr.png')
                    write_image(x_lr[:1, :, :, :], print_res+'/bic_lr.png')
                    write_image(x_hr[:1, :, :, :], print_res+'/bic_hr.png')

                out[0] += x_hr[0]
                return out
