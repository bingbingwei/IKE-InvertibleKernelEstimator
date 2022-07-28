import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.util import *


# noinspection PyUnresolvedReferences
class PixGANLoss(nn.Module):
    """D outputs a [0,1] map of size of the input. This map is compared in a pixel-wise manner to 1/0 according to
    whether the input is real (i.e. from the input image) or fake (i.e. from the Generator)"""

    def __init__(self, d_last_layer_size, in_c=1, mode='original', gamma = None):
        super(PixGANLoss, self).__init__()
        # The loss function is applied after the pixel-wise comparison to the true label (0/1)
        self.loss = nn.L1Loss(reduction='mean')
        # Make a shape
        d_last_layer_shape = [in_c, 1, d_last_layer_size, d_last_layer_size]
        # The two possible label maps are pre-prepared
        self.label_tensor_fake = Variable(torch.zeros(d_last_layer_shape).cuda(), requires_grad=False)
        self.label_tensor_real = Variable(torch.ones(d_last_layer_shape).cuda(), requires_grad=False)
        self.mode = mode
        self.gamma = gamma

    def forward(self, d_last_layer, is_d_input_real):
        # Determine label map according to whether current input to discriminator is real or fake
        label_tensor = self.label_tensor_real if is_d_input_real else self.label_tensor_fake
        opposite_tensor = self.label_tensor_real if not is_d_input_real else self.label_tensor_fake
        # Compute the loss
        # print('GAN loss pred {} \t label {}'.format(d_last_layer.shape, label_tensor.shape))
        if self.mode == 'original':
            return self.loss(d_last_layer, label_tensor)
        elif self.mode == 'focal':
            loss_ce = torch.log(self.loss(d_last_layer, label_tensor))
            loss_op = self.loss(d_last_layer, opposite_tensor)
            return -1 * loss_op**self.gamma * loss_ce


class PixGANSoftmaxLoss(nn.Module):
    """D outputs a [0,1] map of size of the input. This map is compared in a pixel-wise manner to 1/0 according to
    whether the input is real (i.e. from the input image) or fake (i.e. from the Generator)"""

    def __init__(self, d_last_layer_size):
        super(PixGANSoftmaxLoss, self).__init__()
        # The loss function is applied after the pixel-wise comparison to the true label (0/1)
        self.loss = nn.BCELoss(reduction='mean')
        # Make a shape
        d_last_layer_shape = [1, 1, d_last_layer_size, d_last_layer_size]
        # One-hot label maps create
        vec_zero = Variable(torch.zeros(d_last_layer_shape).cuda(), requires_grad=False)
        vec_one = Variable(torch.ones(d_last_layer_shape).cuda(), requires_grad=False)
        # The two possible label maps are pre-prepared; index 0 indicates real while index 1 indicates fake
        self.label_tensor_fake = torch.cat((vec_zero,vec_one), dim=1)
        self.label_tensor_real = torch.cat((vec_one,vec_zero), dim=1)

    def forward(self, d_last_layer, fake_ratio):
        # Determine label map according to whether current input to discriminator is real or fake
        label_tensor = self.label_tensor_real*(1-fake_ratio) + self.label_tensor_fake*fake_ratio
        # Compute the loss
        return self.loss(d_last_layer, label_tensor)


class GANLoss(nn.Module):
    def __init__(self, d_last_layer_size):
        super(GANLoss, self).__init__()
        # The loss function is applied after the pixel-wise comparison to the true label (0/1)
        self.loss = nn.L1Loss(reduction='mean')
        # Make a shape
        d_last_layer_shape = [1, 1, d_last_layer_size, d_last_layer_size]
        # The two possible label maps are pre-prepared
        self.label_tensor_fake = Variable(torch.zeros(d_last_layer_shape).cuda(), requires_grad=False)
        self.label_tensor_real = Variable(torch.ones(d_last_layer_shape).cuda(), requires_grad=False)

    def forward(self, d_last_layer, is_d_input_real):
        # Determine label map according to whether current input to discriminator is real or fake
        label_tensor = self.label_tensor_real if is_d_input_real else self.label_tensor_fake
        return self.loss(d_last_layer, label_tensor)


class EnergyLoss(nn.Module):
    def __init__(self, scale_factor, ksize=19):
        super(EnergyLoss, self).__init__()
        self.loss = nn.L1Loss(reduction='mean')
        # intensity_k = [[1, 2, 1],
        #                [2, 4, 2],
        #                [1, 2, 1]]
        intensity_k = self.get_kernel(ksize, 0.9, 0.9)
        self.intensity_kernel = Variable(intensity_k.float(), requires_grad=False)
        self.scale_factor = scale_factor

    def forward(self, g_input, g_output, shave, mode='patch'):
        g_input_gray = torch.sum(rgb2gray_tensor(g_input), dim=1).unsqueeze(0)
        g_output_gray = torch.sum(rgb2gray_tensor(g_output), dim=1).unsqueeze(0)

        if mode == 'patch':
            downscaled = resize_tensor_w_kernel(im_t=g_input_gray, k=self.intensity_kernel, sf=self.scale_factor)
            downscaled = downscaled[:, :, shave//2:-shave//2, shave//2:-shave//2] if shave != 0 else downscaled
            return self.loss(downscaled, g_output_gray)
        else:
            return self.loss(torch.mean(g_input_gray), torch.mean(g_output_gray))

    def get_kernel(self, ksize, sigmaX, sigmaY):
        kernelX = cv2.getGaussianKernel(ksize, sigmaX)
        kernelY = cv2.getGaussianKernel(ksize, sigmaY)
        return torch.from_numpy(np.outer(kernelY, kernelX.transpose())).cuda()

class DownScaleLoss(nn.Module):
    """ Computes the difference between the Generator's downscaling and an ideal (bicubic) downscaling"""

    def __init__(self, scale_factor, mode='l2'):
        super(DownScaleLoss, self).__init__()

        self.mode = mode
        self.loss_l2 = nn.MSELoss()
        self.loss_l1 = nn.L1Loss()
        bicubic_k = [[0.0001373291015625, 0.0004119873046875, -0.0013275146484375, -0.0050811767578125, -0.0050811767578125, -0.0013275146484375, 0.0004119873046875, 0.0001373291015625],
                     [0.0004119873046875, 0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125, 0.0012359619140625, 0.0004119873046875],
                     [-.0013275146484375, -0.0039825439453130, 0.0128326416015625, 0.0491180419921875, 0.0491180419921875, 0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
                     [-.0050811767578125, -0.0152435302734375, 0.0491180419921875, 0.1880035400390630, 0.1880035400390630, 0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
                     [-.0050811767578125, -0.0152435302734375, 0.0491180419921875, 0.1880035400390630, 0.1880035400390630, 0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
                     [-.0013275146484380, -0.0039825439453125, 0.0128326416015625, 0.0491180419921875, 0.0491180419921875, 0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
                     [0.0004119873046875, 0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125, 0.0012359619140625, 0.0004119873046875],
                     [0.0001373291015625, 0.0004119873046875, -0.0013275146484375, -0.0050811767578125, -0.0050811767578125, -0.0013275146484375, 0.0004119873046875, 0.0001373291015625]]
        self.bicubic_kernel = Variable(torch.Tensor(bicubic_k).cuda(), requires_grad=False)
        self.scale_factor = scale_factor

    def forward(self, g_input, g_output):
        downscaled = resize_tensor_w_kernel(im_t=g_input, k=self.bicubic_kernel, sf=self.scale_factor)

        # Shave the downscaled to fit g_output
        if self.mode == 'l2':
            return self.loss_l2(g_output, shave_a2b(downscaled, g_output))
        elif self.mode == 'l1':
            return self.loss_l1(g_output, shave_a2b(downscaled, g_output))
        elif self.mode == 'l1+l2':
            return 0.5*self.loss_l1(g_output, shave_a2b(downscaled, g_output))+0.5*self.loss_l2(g_output, shave_a2b(downscaled, g_output))

class UpScaleLoss(nn.Module):
    """ Computes the difference between the Generator's downscaling and an ideal (bicubic) downscaling"""

    def __init__(self, scale_factor):
        super(UpScaleLoss, self).__init__()
        self.loss = nn.MSELoss()
        self.scale_factor = scale_factor

    def forward(self, g_input, g_output):
        upscaled = F.interpolate(input=g_input, scale_factor=self.scale_factor, mode='bicubic', align_corners=True)
        # Shave the downscaled to fit g_output
        return self.loss(g_output, shave_a2b(upscaled, g_output))


class SumOfWeightsLoss(nn.Module):
    """ Encourages the kernel G is imitating to sum to 1 """

    def __init__(self):
        super(SumOfWeightsLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(torch.ones(1).to(kernel.device), torch.sum(kernel))


class CentralizedLoss(nn.Module):
    """ Penalizes distance of center of mass from K's center"""

    def __init__(self, k_size, scale_factor=.5):
        super(CentralizedLoss, self).__init__()
        self.indices = Variable(torch.arange(0., float(k_size)).cuda(), requires_grad=False)
        wanted_center_of_mass = k_size // 2 + 0.5 * (int(1 / scale_factor) - k_size % 2)
        self.center = Variable(torch.FloatTensor([wanted_center_of_mass, wanted_center_of_mass]).cuda(), requires_grad=False)
        self.loss = nn.MSELoss()

    def forward(self, kernel):
        """Return the loss over the distance of center of mass from kernel center """
        r_sum, c_sum = torch.sum(kernel, dim=1).reshape(1, -1), torch.sum(kernel, dim=0).reshape(1, -1)
        return self.loss(torch.stack((torch.matmul(r_sum, self.indices) / torch.sum(kernel),
                                      torch.matmul(c_sum, self.indices) / torch.sum(kernel))), self.center)


class BoundariesLoss(nn.Module):
    """ Encourages sparsity of the boundaries by penalizing non-zeros far from the center """

    def __init__(self, k_size):
        super(BoundariesLoss, self).__init__()
        self.mask = map2tensor(create_penalty_mask(k_size, 30))
        self.zero_label = Variable(torch.zeros(k_size).cuda(), requires_grad=False)
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(kernel * self.mask, self.zero_label)


class SparsityLoss(nn.Module):
    """ Penalizes small values to encourage sparsity """
    def __init__(self):
        super(SparsityLoss, self).__init__()
        self.power = 0.2
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(torch.abs(kernel) ** self.power, torch.zeros_like(kernel))


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :]-x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:]-x[:, :, :, :w_x-1]), 2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class InterpolationLoss(nn.Module):
    def __init__(self, scale):
        super(InterpolationLoss,self).__init__()
        self.scale = scale

    def forward(self, lr, sr):
        mask, lr_bic = edge_detect(lr, self.scale)
        loss = nn.L1Loss(reduction='none')(lr_bic, sr)
        return torch.mean((1-mask)*loss)

