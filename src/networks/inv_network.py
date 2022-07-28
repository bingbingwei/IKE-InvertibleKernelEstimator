import torch
import logging
# import modules.discriminator_vgg_arch as SRGAN_arch
from modules.Inv_arch import *
from modules.Subnet_constructor import subnet
import math
import torch.nn as nn

logger = logging.getLogger('base')

scale = 2
subnet_type = 'DBNet'
init = 'xavier'
# block_num = [1]


####################
# define network
####################
def define_G(conf, mode='original'):
    # in_nc = 3 if conf.read_mode == 'RGB' else 1
    # out_nc = 3 if conf.read_mode == 'RGB' else 1
    in_nc = 1
    out_nc = 1
    scale = 1/conf.scale_factor
    if mode == 'original':
        down_num = int(math.log(scale, 2))
        block_num = [conf.block_num]* down_num
        netG = InvRescaleNet(in_nc, out_nc, subnet(subnet_type, init, conf.linear), block_num, down_num, conf.couple_layer,
                             down_sample=conf.downsample, tri_channel=conf.tri_channel, use_res=conf.use_res, down_mode=conf.down_mode, visualize=True)

        netG.output_size = netG.forward(torch.FloatTensor(torch.ones([1, in_nc, conf.input_crop_size, conf.input_crop_size]))).shape[-1]
        netG.forward_shave = int(conf.input_crop_size * conf.scale_factor) - netG.output_size
    return netG
