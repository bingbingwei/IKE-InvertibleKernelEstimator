import torch.nn as nn
from utils.util import *
import yaml
from utils.kernel_downsampler import *
from modules.Quantization import Quantization


class Peeker:
    def __init__(self, conf, tb_logger):
        self.conf = conf
        self.tb_logger = tb_logger
        self.iter = 0
        self.channel = 3 if self.conf.read_mode == 'RGB' else 1
        self.Quantization = Quantization()

    def peek(self, input, netG, netU, task_id):
        # Get Peek Image
        with torch.no_grad():
            with open(self.conf.peek_yaml_path, 'r') as stream:
                info = yaml.load(stream, Loader=yaml.FullLoader)
            kernel = get_kernel(info['ksize'], info['sigmaX'], info['sigmaY'])

            # LQ version for KernelGAN evaluation
            src = input.squeeze(0).permute(1, 2, 0).cpu().numpy()
            self.peek_image = torch.from_numpy(downsample(src, kernel, info['inter'])).cuda()

            if self.conf.INN_mode:
                out = netG.forward(input)
                g_output = out[:1, :, :, :]
                z_output = out[1:, :, :, :]
            else:
                g_output = netG.forward(input)
                # Invertible mode will be the same size as original image while KernelGAN will not
                # Shave the extra part for KernelGAN
                shave = netG.forward_shave // 2
                self.peek_image = self.peek_image[shave:-shave, shave:-shave, :]

            self.peek_image = self.peek_image.permute(2, 0, 1).unsqueeze(0)
            loss_lq = nn.L1Loss()(self.peek_image, g_output)

            self.tb_logger.add_scalar('Task_{}/Peek'.format(task_id), loss_lq.item(), self.iter)

            # HQ version for KernelGAN evaluation
            if self.conf.peekHQ:
                if self.conf.read_mode == 'HSV':
                    src = read_image(self.conf.eval_image_path, self.conf.read_mode)[:, :, -self.channel:] / 255
                    src_lr = read_image(self.conf.input_image_path, self.conf.read_mode)[:, :, -self.channel:] / 255
                else:
                    src = read_image(self.conf.eval_image_path, self.conf.read_mode)[:, :, :self.channel] / 255
                    src_lr = read_image(self.conf.input_image_path, self.conf.read_mode)[:, :, -self.channel:] / 255

                src = im2tensor(src)
                src_lr = im2tensor(src_lr)
                # Generate peek image hq if this is the first iteration

                gth = src.squeeze(0).permute(1, 2, 0).cpu().numpy()
                peek_image_hq = torch.from_numpy(downsample(gth, kernel, info['inter'])).cuda()
                if not self.conf.INN_mode:
                    # Invertible mode will be the same size as original image while KernelGAN will not
                    # Shave the extra part for KernelGAN
                    shave =netG.forward_shave // 2
                    peek_image_hq = peek_image_hq[shave:-shave, shave:-shave, :]
                peek_image_hq = peek_image_hq.permute(2, 0, 1).unsqueeze(0)

                if self.conf.INN_mode:
                    out = netG.forward(src)
                    out_lr = netG.forward(src_lr)
                    z_pred = netU.forward(out_lr)
                    g_output = out[:1, :, :, :]
                    z_output = out[1:, :, :, :]
                    loss_z = nn.L1Loss()(z_pred, z_output)

                    self.tb_logger.add_scalar('Task_{}/Peek Z'.format(task_id), loss_z.item(), self.iter)
                else:
                    g_output = netG.forward(src)

                loss_hq = nn.L1Loss()(peek_image_hq, g_output)

                self.tb_logger.add_image('Peek/Gth_{}'.format(task_id), tensor2tb_log(peek_image_hq[0]),
                                         self.iter)
                self.tb_logger.add_scalar('Task_{}/Peek HQ'.format(task_id), loss_hq.item(), self.iter)


                # Color shifting debug
                if self.conf.read_mode == 'RGB':
                    red_dict, blue_dict, green_dict = {}, {}, {}
                    red_dict['HQ'], blue_dict['HQ'], green_dict['HQ'] = torch.mean(src, (0, 2, 3))
                    red_dict['LQ'], blue_dict['LQ'], green_dict['LQ'] = torch.mean(peek_image_hq, (0, 2, 3))
                    red_dict['Pred'], blue_dict['Pred'], green_dict['Pred'] = torch.mean(g_output, (0, 2, 3))
                    self.tb_logger.add_scalars('Color Shift Debug/Red_{}'.format(task_id), red_dict, self.iter)
                    self.tb_logger.add_scalars('Color Shift Debug/Blue_{}'.format(task_id), blue_dict, self.iter)
                    self.tb_logger.add_scalars('Color Shift Debug/Green_{}'.format(task_id), green_dict, self.iter)
            self.iter += 1

    def peek_final(self, netG):
        with torch.no_grad():
            img_LR = read_image(self.conf.input_image_path,self.conf.read_mode)[:,:,:self.channel]/255.0
            img_LR = im2tensor((img_LR))

            out = netG.forward(img_LR)
            g_output = out[:1, :, :, :]
            z_output = out[1:, :, :, :]
            g_quant = self.Quantization(g_output)

            input_inv = torch.cat((g_quant, z_output), 0)
            img_SR = netG.forward(input_inv, rev=True)
            write_image(img_SR, self.conf.output_img_path+'/Quant_LR.png')
            return

    def peek_z(self, netG, netU, task_id):
        with torch.no_grad():
            src = read_image(self.conf.eval_image_path, self.conf.read_mode) / 255
            src_lr = read_image(self.conf.input_image_path, self.conf.read_mode) / 255
            src = im2tensor(src)
            src_lr = im2tensor(src_lr)
            out = netG.forward(src)
            out_lr = netG.forward(src_lr)
            z_pred = netU.forward(out_lr[1:,:,:,:])
            g_output = out[:1, :, :, :]
            z_output = out[1:, :, :, :]
            loss_z = nn.L1Loss()(z_pred, z_output)

            self.tb_logger.add_scalar('Task_{}/Peek Z'.format(task_id), loss_z.item(), self.iter)
            self.iter += 1