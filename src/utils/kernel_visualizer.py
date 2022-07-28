from networks import network
import torch
from utils.util import read_image, im2tensor, post_process_k, write_image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import loss


def calc_curr_k(G):
    """given a generator network, the function calculates the kernel it is imitating"""
    delta = torch.Tensor([1.]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
    for ind, w in enumerate(G.parameters()):
        curr_k = F.conv2d(delta, w, padding=12) if ind == 0 else F.conv2d(curr_k, w)
    curr_k = curr_k.squeeze().flip([0, 1])

    return  curr_k


def show_kernel(conf, gan, iters, tb_logger, hr_path, lr_path=None):
    # init
    G = network.Generator(conf).cuda()
    G.apply(network.weights_init_G)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=conf.g_lr, betas=(conf.beta1, 0.999))

    src_hr = im2tensor(read_image(hr_path))
    if lr_path == None:
        with torch.no_grad():
            src_lr = gan.G.forward(src_hr)[:1, :, :, :]

    # range for src_lr
    crop_size = 64

    for i in range(iters):
        optimizer_G.zero_grad()
        top, left = np.random.choice(range(src_lr.shape[2]-crop_size)), np.random.choice(range(src_lr.shape[3]-crop_size))
        top_hr, left_hr = 2*top, 2*left
        pred_lr = G.forward(src_hr[:,:,top_hr:top_hr+crop_size*2, left_hr:left_hr+crop_size*2])
        dst_lr = src_lr[:, :, top+3:top+crop_size-3, left+3:left+crop_size-3]
        loss = nn.L1Loss(reduction='mean')(pred_lr, dst_lr)
        kernel = calc_curr_k(G)
        loss += calc_constraints(kernel)
        print(loss.item())
        loss.backward()
        tb_logger.add_scalar('Kernel/Task', loss.item(), i)
        optimizer_G.step()

    kernel = calc_curr_k(G)
    kernel = post_process_k(kernel, 10)

    with torch.no_grad():
        pred_lr = G.forward(src_hr)
        write_image(pred_lr, 'lr.png')
        write_image(src_lr, 'src_lr.png')

    sio.savemat('kernel.mat', {'Kernel': kernel})

def calc_constraints(kernel):
    # Calculate constraints
    sum2one_loss = loss.SumOfWeightsLoss().cuda()
    boundaries_loss = loss.BoundariesLoss(k_size=13).cuda()
    centralized_loss = loss.CentralizedLoss(k_size=13, scale_factor=0.5).cuda()
    sparse_loss = loss.SparsityLoss().cuda()
    loss_boundaries = boundaries_loss.forward(kernel=kernel)
    loss_sum2one = sum2one_loss.forward(kernel=kernel)
    loss_centralized = centralized_loss.forward(kernel=kernel)
    loss_sparse = sparse_loss.forward(kernel=kernel)
    # Apply constraints co-efficients
    return loss_sum2one * 0.5 + \
           loss_boundaries * 0.5 + loss_centralized * 1 + \
           loss_sparse * 5