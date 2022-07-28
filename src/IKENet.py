import loss
from networks import network, inv_network
from networks.Zupsamplers.RCAN import RCAN
from networks.Zupsamplers.EDSR import EDSR
from utils.util import *
import yaml
from utils.kernel_downsampler import *
from modules.Quantization import Quantization
from networks.Upsampler import GANupsampler, LLRupsampler
from torch.autograd import Variable

class KernelGAN:
    # Constraint co-efficients
    lambda_sum2one = 0.5
    lambda_bicubic = 5
    lambda_boundaries = 0.5
    lambda_centralized = 0
    lambda_sparse = 0
    lambda_image_discriminate = 5
    lambda_energy = 0
    lambda_tv = 1
    lambda_3lr = 10
    lamda_cycleGAN = 2
    lambda_inter = 5

    gd_ratio = 5

    start_image_discriminate = False
    start_z_up_train = False

    def __init__(self, conf, tb_logger, task_id):
        # Acquire configuration
        self.conf = conf

        # State Channel

        self.channel = 3 if self.conf.read_mode == 'RGB' else 1
        self.scale = int(1/conf.scale_factor)
        self.crop_size = conf.input_crop_size

        # Define the GAN
        self.G = inv_network.define_G(conf, mode='original').cuda()

        self.G.apply(network.weights_init_G)

        self.D = network.Discriminator(conf).cuda()
        self.D.apply(network.weights_init_D)

        self.UD = network.Discriminator(conf).cuda()
        self.UD.apply(network.weights_init_D)

        self.D_image = network.ImageDiscriminator(conf).cuda()
        self.D_image.apply(network.weights_init_D)

        self.D_res = network.Discriminator(conf).cuda()
        self.D_res.apply(network.weights_init_D)

        print(self.conf.z_upsample)
        if self.conf.z_upsample == 'origin':
            self.U = network.Upsample(conf).cuda()
        elif self.conf.z_upsample == 'RCAN':
            self.U = RCAN(conf=conf).cuda()
        elif self.conf.z_upsample == 'EDSR':
            self.U = EDSR(conf=conf).cuda()
        elif self.conf.z_upsample == 'ZSSR':
            self.U = network.ZSSRUpsampler(conf=conf, input_channels=int(3*(1/(self.conf.scale_factor**2)-1)))

        self.U.apply(network.weights_init_G)

        # Optimizers
        if conf.optim == 'adam':
            self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=conf.g_lr, betas=(conf.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=conf.d_lr, betas=(conf.beta1, 0.999))
            self.optimizer_D_image = torch.optim.Adam(self.D_image.parameters(), lr=conf.d_lr, betas=(conf.beta1, 0.999))
            self.optimizer_UD = torch.optim.Adam(self.UD.parameters(), lr=conf.d_lr, betas=(conf.beta1, 0.999))


            self.optimizer_U = torch.optim.Adam(self.U.parameters(), lr=conf.u_lr, betas=(conf.beta1, 0.999))
            self.optimizer_D_res = torch.optim.Adam(self.D_res.parameters(), lr=conf.d_lr, betas=(conf.beta1, 0.999))


        elif conf.optim == 'SGD':
            self.optimizer_G = torch.optim.SGD(self.G.parameters(), lr=conf.g_lr, momentum=0.9)
            self.optimizer_D = torch.optim.SGD(self.D.parameters(), lr=conf.d_lr, momentum=0.9)
            self.optimizer_D_image = torch.optim.SGD(self.D_image.parameters(), lr=conf.d_lr, momentum=0.9)
            self.optimizer_UD = torch.optim.SGD(self.UD.parameters(), lr=conf.d_lr, momentum=0.9)


            self.optimizer_U = torch.optim.SGD(self.U.parameters(), lr=conf.u_lr, momentum=0.9)


        # Calculate D's input & output shape according to the shaving done by the networks
        self.d_input_shape = self.G.output_size
        self.d_output_shape = self.d_input_shape - self.D.forward_shave

        # Input tensors
        self.g_input = torch.FloatTensor(1, self.channel, conf.input_crop_size, conf.input_crop_size).cuda()
        self.d_input = torch.FloatTensor(1, self.channel, self.d_input_shape, self.d_input_shape).cuda()

        # The kernel G is imitating
        self.curr_k = torch.FloatTensor(conf.G_kernel_size, conf.G_kernel_size).cuda()

        # Losses
        if not conf.mixup:
            if 'focal' in conf.other_loss:
                self.GAN_loss_layer = loss.PixGANLoss(d_last_layer_size=self.d_output_shape, mode='focal', gamma=5).cuda()
            else:
                self.GAN_loss_layer = loss.PixGANLoss(d_last_layer_size=self.d_output_shape).cuda()
        else:
            self.GAN_loss_layer = loss.PixGANSoftmaxLoss(d_last_layer_size=self.d_output_shape).cuda()

        self.loss_bicubic = 0

        # Define loss function
        self.criterionGAN = self.GAN_loss_layer.forward
        self.criterionGAN_ud = loss.PixGANLoss(d_last_layer_size=self.crop_size-6).cuda()

        # TensorBoard Logger
        self.iter = 0
        self.task_id = task_id
        self.tb_logger = tb_logger

        # Peek and visual Image
        self.peek_image = None
        self.g_pred_lst = []
        self.z1_pred_lst, self.z2_pred_lst, self.z3_pred_lst = [], [], []
        self.z1_up_lst, self.z2_up_lst, self.z3_up_lst = [], [], []
        self.z_pred_lst = []
        self.z_up_lst = []
        self.d_real_lst = []
        self.d_fake_lst = []
        self.peekHQ_lst = []
        self.peek_image_hq = None

        # Upsampler for Z
        self.upsamples = []
        self.up_mode = []
        self.gan_upsample = 'gan' in conf.z_upsample

        self.U_GAN = None
        self.U_LLR = None
        if self.gan_upsample:
            self.U_GAN = GANupsampler(conf, tb_logger, task_id)

        self.U_LLR = LLRupsampler(conf, task_id, self.U, self.tb_logger).cuda()

        for mode in conf.z_upsample:
            if self.gan_upsample:
                self.upsamples.append(self.U_GAN.forward)
            else:
                self.upsamples.append(self.U)

            self.up_mode.append(mode)

        self.bicubic_loss = loss.DownScaleLoss(scale_factor=conf.scale_factor, mode=conf.bicubic_loss).cuda()
        self.energy_loss = loss.EnergyLoss(scale_factor=conf.scale_factor, ksize=conf.energy_k_size).cuda()
        self.cycle_loss = nn.L1Loss(reduction='mean')
        self.tv_loss = loss.TVLoss().cuda()
        self.interloss = loss.InterpolationLoss(1/conf.scale_factor).cuda()
        self.lllr_loss = self.U_LLR

        self.Quantization = Quantization()

        print('*' * 60 + '\nSTARTED KernelGAN on: \"%s\"...' % conf.input_image_path)

    def calc_curr_k(self):
        """given a generator network, the function calculates the kernel it is imitating"""
        delta = torch.Tensor([1.]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
        for ind, w in enumerate(self.G.parameters()):
            curr_k = F.conv2d(delta, w, padding=self.conf.G_kernel_size - 1) if ind == 0 else F.conv2d(curr_k, w)
        self.curr_k = curr_k.squeeze().flip([0, 1])

    def train(self, g_input, d_input, r_input=None):
        self.set_input(g_input, d_input, r_input)

        out = self.train_g()
        self.train_d()

        self.tb_logger.flush()
        self.iter += 1

    def set_input(self, g_input, d_input, r_input=None):
        if self.conf.read_mode == 'HSV':
            self.g_input = g_input.contiguous()[:, -self.channel:, :, :]
            self.d_input = d_input.contiguous()[:, -self.channel:, :, :]
            self.g_sv = g_input.contiguous()[:, :-self.channel, :, :]
            self.d_sv = d_input.contiguous()[:, :-self.channel, :, :]
        else:
            self.g_input = g_input.contiguous()[:, :self.channel, :, :]
            self.d_input = d_input.contiguous()[:, :self.channel, :, :]

    def train_g(self):
        # Zeroize gradients
        self.optimizer_G.zero_grad()

        # Generator forward pass
        out = self.G.forward(self.g_input)

        g_pred = out[:1, :, :, :]
        z_pred = out[1:, :, :, :]

        # Pass Generators output through Discriminator
        d_pred_fake = self.D.forward(g_pred)

        # Calculate generator loss, based on discriminator prediction on generator result
        if not self.conf.mixup:
            loss_g = self.criterionGAN(d_last_layer=d_pred_fake, is_d_input_real=True)
        else:
            loss_g = self.criterionGAN(d_last_layer=d_pred_fake, fake_ratio=0)

        self.loss_bicubic = self.bicubic_loss.forward(g_input=self.g_input, g_output=g_pred)
        extra_constrain = self.loss_bicubic * self.lambda_bicubic

        if 'image_discriminate' in self.conf.other_loss and self.start_image_discriminate:
            d_image_fake = self.D_image.forward(g_pred)
            label = Variable(torch.ones(1).cuda(), requires_grad=False)
            loss_d_image = nn.L1Loss(reduction='mean')(d_image_fake, label)
            extra_constrain += self.lambda_image_discriminate*loss_d_image

        self.loss_energy = self.energy_loss.forward(g_input=self.g_input, g_output=g_pred, shave=self.G.forward_shave)
        if 'energy' in self.conf.other_loss:
            extra_constrain += self.loss_energy * self.lambda_energy
        if 'energy_all' in self.conf.other_loss:
            loss_energy_all = self.energy_loss.forward(g_input=self.g_input, g_output=g_pred, shave=self.G.forward_shave, mode='all')
            extra_constrain += loss_energy_all * self.lambda_energy


        # Sum all losses
        total_loss_g = loss_g + extra_constrain
        # Calculate gradients
        total_loss_g.backward(retain_graph=True)
        # Update weights
        self.optimizer_G.step()

        # if self.start_z_up_train:
        # self.train_u(out.detach())

        # TensorBoard Logger add scalar
        self.tb_logger.add_scalar('Task_{}/Loss G'.format(self.task_id), loss_g.item(), self.iter)
        self.tb_logger.add_scalar('Task_{}/Loss bicubic'.format(self.task_id), self.loss_bicubic.item(), self.iter)
        self.tb_logger.add_scalar('Task_{}/Lambda energy'.format(self.task_id), self.lambda_energy, self.iter)
        self.tb_logger.add_scalar('Task_{}/Other constrain'.format(self.task_id), (extra_constrain-self.loss_bicubic * self.lambda_bicubic).item(), self.iter)
        self.tb_logger.add_scalar('Task_{}/Loss Energy'.format(self.task_id), self.loss_energy.item(), self.iter)

        # Z analysis
        self.tb_logger.add_scalar('Z_Analysis/Mean_{}'.format(self.conf.filename), torch.mean(z_pred).item(), self.iter)
        self.tb_logger.add_scalar('Z_Analysis/Var_{}'.format(self.conf.filename), torch.var(z_pred).item(), self.iter)

        return out.detach()

    def train_u(self, out, cal_psnr=False, img=None, freeze=False):
        # Z module params update
        self.optimizer_U.zero_grad()
        if not freeze:
             self.optimizer_G.zero_grad()
        else:
            detach_network(self.G)
            detach_network(self.UD)

        z_up_constrain = torch.zeros((1)).cuda()
        keys = ['cycle_consistency', 'tv_loss', '3lr', 'interpolation_loss']
        if check_key_in_dict(keys, self.conf.other_loss):

            z_inv = self.U.forward(out[1:,:,:,:])
            input_inv = torch.cat((self.g_input, z_inv), 0)

            img_SR = self.G.forward(input_inv, rev=True)
            if 'interpolation_loss' in self.conf.other_loss:
                loss_inter = self.interloss.forward(self.g_input, img_SR)
                z_up_constrain += self.lambda_inter * loss_inter

            if 'cycle_consistency' in self.conf.other_loss:
                pred = self.UD.forward(img_SR)
                label_tensor_true = Variable(torch.ones(pred.shape).cuda(), requires_grad=False)
                loss_cycle = self.cycle_loss(pred, label_tensor_true)
                loss_cycle_energy = self.energy_loss.forward(g_input=img_SR, g_output=self.g_input,
                                                             shave=self.G.forward_shave)
                z_up_constrain += loss_cycle*self.lamda_cycleGAN # + loss_cycle_energy * self.lambda_energy

            if 'tv_loss' in self.conf.other_loss:
                loss_tv = self.tv_loss.forward(img_SR)
                z_up_constrain += self.lambda_tv * loss_tv
            if '3lr' in self.conf.other_loss:
                loss_3lr = self.lllr_loss(self.g_input, self.G)
                z_up_constrain += self.lambda_3lr*loss_3lr

            if self.conf.residual_supervised:
                upsampler = nn.Upsample(scale_factor=1/self.conf.scale_factor, mode='bicubic', align_corners=True)
                pred = self.D_res.forward(img_SR-upsampler(self.g_input))
                label_tensor_true = Variable(torch.ones(pred.shape).cuda(), requires_grad=False)
                loss_cycle = self.cycle_loss(pred, label_tensor_true)
                z_up_constrain += loss_cycle

            # Visualization of Zs
            if (self.iter+1) % 100 == 0:
                self.visual_z(out, z_inv)

        z_up_constrain.backward()
        self.optimizer_U.step()

        if not freeze:
            self.optimizer_G.step()

        else:
            attach_network(self.G)
            attach_network(self.UD)

        if 'cycle_consistency' in self.conf.other_loss:
            self.tb_logger.add_scalar('Task_{}/Loss CycleGAN'.format(self.task_id), loss_cycle.item(), self.iter)

        if 'tv_loss' in self.conf.other_loss:
            self.tb_logger.add_scalar('Task_{}/Loss TV'.format(self.task_id), loss_tv.item(), self.iter)

        if '3lr' in self.conf.other_loss:
            self.tb_logger.add_scalar('Task_{}/Loss U'.format(self.task_id), loss_3lr.item(), self.iter)

        if 'interpolation_loss' in self.conf.other_loss:
            self.tb_logger.add_scalar('Task_{}/Loss inter'.format(self.task_id), loss_inter.item(), self.iter)

        # Calculate PSNR for early stopping analysis
        if cal_psnr:
            img_SR = self.test(write_result=False)
            img_HR = read_image(self.conf.eval_image_path, self.conf.read_mode)[:, :, :self.channel]/255.0
            img_HR = im2tensor((img_HR))
            self.tb_logger.add_scalar('PSNR/{}'.format(self.conf.filename), calculate_psnr(img_SR, img_HR), self.iter)

        # Z analysis
        self.tb_logger.add_scalar('Z_Analysis/Mean_{}'.format(self.conf.filename), torch.mean(out[1:,:,:,:]).item(), self.iter)
        self.tb_logger.add_scalar('Z_Analysis/Var_{}'.format(self.conf.filename), torch.var(out[1:,:,:,:]).item(), self.iter)

        self.tb_logger.add_scalar('Z_Analysis/Upsample_Mean_{}'.format(self.conf.filename), torch.mean(z_inv).item(), self.iter)
        self.tb_logger.add_scalar('Z_Analysis/Upsample_Var_{}'.format(self.conf.filename), torch.var(z_inv).item(), self.iter)

    def train_d(self):
        # Zeroize gradients
        self.optimizer_D.zero_grad()

        # Discriminator forward pass over real example
        # Discriminator forward pass over fake example (generated by generator)
        # Note that generator result is detached so that gradients are not propagating back through generator
        g_output = self.G.forward(self.g_input)[:1, :, :, :]

        # Calculate discriminator loss
        if not self.conf.mixup:
            d_pred_real, feat_real = self.D.forward(self.d_input, True)
            d_pred_fake, feat_fake = self.D.forward((g_output + torch.randn_like(g_output) / 255.).detach(), True)
            loss_d_fake = self.criterionGAN(d_pred_fake, is_d_input_real=False)
            loss_d_real = self.criterionGAN(d_pred_real, is_d_input_real=True)
            loss_d = (loss_d_fake + loss_d_real) * 0.5

        else:
            # random a ratio between 0.3~0.7
            fake_ratio = ((0.7-0.3)*torch.rand(1, requires_grad=False)+0.3).cuda()
            img_mixup = g_output*fake_ratio + self.d_input*(1-fake_ratio)
            d_pred_mixup = self.D.forward(img_mixup)
            loss_d = self.criterionGAN(d_pred_mixup, fake_ratio=fake_ratio)

        if 'image_discriminate' in self.conf.other_loss and self.start_image_discriminate:
            self.optimizer_D_image.zero_grad()
            loss_d_img = 0.5*self.train_d_img(g_output.detach(), False)
            loss_d_img += 0.5*self.train_d_img(self.d_input, True)
            loss_d_img.backward()
            self.optimizer_D_image.step()
            self.tb_logger.add_scalar('Task_{}/Loss D image'.format(self.task_id), loss_d_img.item(), self.iter)

        # Calculate gradients, note that gradients are not propagating back through generator
        loss_d.backward()
        # Update weights, note that only discriminator weights are updated (by definition of the D optimizer)
        self.optimizer_D.step()

        # TensorBoard Logger add scalar
        self.tb_logger.add_scalar('Task_{}/Loss D'.format(self.task_id), loss_d.item(), self.iter)

    def train_ud(self, out, mode='all'):
        # Zeroize gradients
        self.optimizer_UD.zero_grad()

        z_inv = self.U.forward(out[1:, :, :, :])
        input_inv = torch.cat((self.g_input, z_inv), 0)
        img_SR = self.G.forward(input_inv, rev=True)

        d_pred_real, feat_real = self.UD.forward(self.g_input, True)
        loss_d_real = self.criterionGAN_ud(d_pred_real, is_d_input_real=True)
        # ALL MODE
        loss_d_fake = torch.zeros((1)).cuda()
        for  i in range(self.scale):
            for j in range(self.scale):
                d_pred_fake, feat_fake = self.UD.forward(img_SR[:, :, i*self.crop_size:(i+1)*self.crop_size, j*self.crop_size:(j+1)*self.crop_size], True)
                loss_d_fake += self.criterionGAN_ud(d_pred_fake, is_d_input_real=False)
        loss_d_fake /= (self.scale**2)

        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()
        self.optimizer_UD.step()
        #
        # # TensorBoard Logger add scalar
        self.tb_logger.add_scalar('Task_{}/Loss UD'.format(self.task_id), loss_d.item(), self.iter)

    # Only in Invertible mode
    def test(self, write_result=True):
        assert self.conf.INN_mode
        with torch.no_grad():
            img_LR = read_image(self.conf.input_image_path,self.conf.read_mode)[:,:,:self.channel]/255.0
            img_LR = im2tensor((img_LR))

            out = self.G.forward(img_LR)

            z = out[1:, :, :, :]
            z_np = move2cpu(z)
            np.save(self.conf.np_file_path+'/z.npy', z_np)

            z_inv = self.U(z)
            input_inv = torch.cat((img_LR, z_inv), 0)

            # z_inv_np = move2cpu(z_inv)
            # np.save(self.conf.np_file_path+'/z_up.npy', z_inv_np)

            img_SR = self.G.forward(input_inv, rev=True, print_res=self.conf.output_img_path)
            if write_result:
                self.tb_logger.add_image('Pred_{}/Final HR'.format(self.task_id), tensor2tb_log(img_SR[0]), self.iter)
                self.tb_logger.add_image('Pred_{}/Final LR'.format(self.task_id), tensor2tb_log(out[0, :, :, :]), self.iter)

                write_image(img_SR, self.conf.output_img_path+'/SR_{}_{}.png'.format(self.conf.couple_layer, self.conf.downsample))
                write_image(out[:1, :, :, :], self.conf.output_img_path+'/LR_{}_{}.png'.format(self.conf.couple_layer, self.conf.downsample))
            else:
                self.tb_logger.add_scalar('Z_Analysis_whole/Upsample_Mean_{}'.format(self.conf.filename),
                                          torch.mean(z_inv).item(), self.iter)
                self.tb_logger.add_scalar('Z_Analysis_whole/Upsample_Var_{}'.format(self.conf.filename),
                                          torch.var(z_inv).item(), self.iter)
                self.tb_logger.add_scalar('Z_Analysis_whole/Mean_{}'.format(self.conf.filename),
                                          torch.mean(z).item(), self.iter)
                self.tb_logger.add_scalar('Z_Analysis_whole/Var_{}'.format(self.conf.filename),
                                          torch.var(z).item(), self.iter)

                return img_SR

    def test_x4(self):
        assert self.conf.INN_mode
        with torch.no_grad():
            img_LR = read_image(self.conf.input_image_path,self.conf.read_mode)[:,:,:self.channel]/255.0
            img_LR = im2tensor((img_LR))

            out = self.G.forward(img_LR)

            z_inv = self.U(out[1:, :, :, :])
            input_inv = torch.cat((img_LR, z_inv), 0)

            img_SR_semi = self.G.forward(input_inv, rev=True, print_res=self.conf.output_img_path)

            z_inv = self.U(z_inv)
            input_inv = torch.cat((img_SR_semi, z_inv), 0)
            img_SR = self.G.forward(input_inv, rev=True, print_res=self.conf.output_img_path)


            self.tb_logger.add_image('Pred_{}/Final HR'.format(self.task_id), tensor2tb_log(img_SR[0]), self.iter)
            self.tb_logger.add_image('Pred_{}/Final LR'.format(self.task_id), tensor2tb_log(out[0, :, :, :]), self.iter)

            write_image(img_SR, self.conf.output_img_path+'/SR_{}_{}.png'.format(self.conf.couple_layer, self.conf.downsample))
            write_image(out[:1, :, :, :], self.conf.output_img_path+'/LR_{}_{}.png'.format(self.conf.couple_layer, self.conf.downsample))

    # visualize the visualize crop
    def visual(self, g_input, d_input):
        g_input = g_input[:, :self.channel, :, :]
        d_input = d_input[:, :self.channel, :, :]
        with torch.no_grad():
            out = self.G.forward(g_input)
            g_pred = out[:1, :, :, :]
            z_pred = out[1:, :, :, :]
            z_vis = tensor2tb_log(z_pred[:1,:,:,:].squeeze(0))
            self.z_pred_lst.append(z_vis)
            if self.gan_upsample:
                z_up = self.U_GAN.forward(z_pred)
                z_up_vis = tensor2tb_log(z_up[:1,:,:,:].squeeze(0))
                self.z_up_lst.append(z_up_vis)

            d_pred_fake = self.D.forward(g_pred)
            d_pred_real = self.D.forward(d_input)

            g_pred = tensor2tb_log(g_pred.squeeze(0))

            d_pred_real = d_pred_real.squeeze(0)
            d_pred_fake = d_pred_fake.squeeze(0)

            if self.conf.mixup:
                d_pred_real = d_pred_real[0].unsqueeze(0)
                d_pred_fake = d_pred_fake[0].unsqueeze(0)

            d_pred_real = tensor2tb_log(d_pred_real)
            d_pred_fake = tensor2tb_log(d_pred_fake)

            self.g_pred_lst.append(g_pred)
            self.d_real_lst.append(d_pred_real)
            self.d_fake_lst.append(d_pred_fake)

    def finish(self):
        if self.conf.INN_test:
            self.test()
            if self.conf.eval:
                img_SR = cv2.imread(self.conf.output_img_path+'/{}_SR_{}_{}.png'.format(self.conf.filename, self.conf.couple_layer, self.conf.downsample))
                img_HR = cv2.imread(self.conf.eval_image_path)
                psnr = calculate_psnr(img_SR, img_HR)
                print('Super Resolution Task Finished with PSNR {}'.format(psnr))
                self.tb_logger.add_scalar('PSNR/Task', psnr, self.task_id)
                self.tb_logger.flush()

        if len(self.z1_pred_lst) != 0:
            self.z1_pred_lst = np.expand_dims(np.array(self.z1_pred_lst), axis=0)
            self.tb_logger.add_video('Pred/Z1 {}'.format(self.task_id), self.z1_pred_lst, self.iter)
        if len(self.z1_up_lst) != 0:
            self.z1_up_lst = np.expand_dims(np.array(self.z1_up_lst), axis=0)
            self.tb_logger.add_video('Pred/Z1 up {}'.format(self.task_id), self.z1_up_lst, self.iter)
        if len(self.z2_pred_lst) != 0:
            self.z2_pred_lst = np.expand_dims(np.array(self.z2_pred_lst), axis=0)
            self.tb_logger.add_video('Pred/Z2 {}'.format(self.task_id), self.z2_pred_lst, self.iter)
        if len(self.z2_up_lst) != 0:
            self.z2_up_lst = np.expand_dims(np.array(self.z2_up_lst), axis=0)
            self.tb_logger.add_video('Pred/Z2 up {}'.format(self.task_id), self.z2_up_lst, self.iter)
        if len(self.z3_pred_lst) != 0:
            self.z3_pred_lst = np.expand_dims(np.array(self.z3_pred_lst), axis=0)
            self.tb_logger.add_video('Pred/Z3 {}'.format(self.task_id), self.z3_pred_lst, self.iter)
        if len(self.z3_up_lst) != 0:
            self.z3_up_lst = np.expand_dims(np.array(self.z3_up_lst), axis=0)
            self.tb_logger.add_video('Pred/Z3 up {}'.format(self.task_id), self.z3_up_lst, self.iter)
        if len(self.d_real_lst) != 0:
            self.d_real_lst = np.expand_dims(np.array(self.d_real_lst), axis=0)
            self.tb_logger.add_video('D Real/img_{}'.format(self.task_id), self.d_real_lst, self.iter)
        if len(self.d_fake_lst) != 0:
            self.d_fake_lst = np.expand_dims(np.array(self.d_fake_lst), axis=0)
            self.tb_logger.add_video('D Fake/img_{}'.format(self.task_id), self.d_fake_lst, self.iter)
        if len(self.peekHQ_lst) != 0:
            self.peekHQ_lst = np.expand_dims(np.array(self.peekHQ_lst), axis=0)
            self.tb_logger.add_video('Peek/imgs {}'.format(self.task_id), self.peekHQ_lst, self.iter)
            self.tb_logger.add_image('Peek/img {}'.format(self.task_id), self.peekHQ_lst[:, -1].squeeze(0), self.iter)
        self.tb_logger.flush()

        print('*' * 60 + '\n')


    def save_result(self):
        src = read_image(self.conf.eval_image_path, self.conf.read_mode)[:, :, :self.channel] / 255
        src = im2tensor(src)
        g_output = self.G.forward(src)[:1, :, :, :]
        write_image(g_output, self.conf.output_img_path + '/INN.png')

    def test_hsv(self, output):
        with torch.no_grad():
            down = nn.Upsample(scale_factor=self.conf.scale_factor, mode='nearest')
            input = read_image(self.conf.eval_image_path,self.conf.read_mode)[:,:,:-self.channel] / 255
            input = im2tensor(input)
            input = down(input)

            out = torch.cat((input, output), 1)
            out = tensor2im(out)
            out = hsv2rgb(out)

            with open(self.conf.peek_yaml_path, 'r') as stream:
                info = yaml.load(stream, Loader=yaml.FullLoader)
            kernel = get_kernel(info['ksize'], info['sigmaX'], info['sigmaY'])

            src = read_image(self.conf.eval_image_path, 'RGB') / 255
            src = im2tensor(src)
            # Generate peek image hq if this is the first iteration
            gth = src.squeeze(0).permute(1, 2, 0).cpu().numpy()
            peek_image_hq = torch.from_numpy(downsample(gth, kernel, info['inter'])).cuda()
            peek_image_hq = peek_image_hq.permute(2, 0, 1).unsqueeze(0)

            loss_hq = nn.L1Loss()(peek_image_hq, im2tensor(out))

            self.tb_logger.add_scalar('Task_{}/Peek HQ HSV'.format(self.task_id), loss_hq.item(), self.iter)
            self.tb_logger.add_image('Peek/HSV {}'.format(self.task_id), tensor2tb_log(im2tensor(out).squeeze(0)), self.iter)

    def visual_z(self, z_lr, z_sr):
        z1_lr_vis = tensor2tb_log(z_lr[1, :, :, :])
        z2_lr_vis = tensor2tb_log(z_lr[2, :, :, :])
        z3_lr_vis = tensor2tb_log(z_lr[3, :, :, :])
        # self.z1_pred_lst.append(z1_lr_vis)
        # self.z2_pred_lst.append(z2_lr_vis)
        # self.z3_pred_lst.append(z3_lr_vis)
        self.tb_logger.add_image('Peek_{}/z1'.format(self.task_id), z1_lr_vis, self.iter)
        self.tb_logger.add_image('Peek_{}/z2'.format(self.task_id), z2_lr_vis, self.iter)
        self.tb_logger.add_image('Peek_{}/z3'.format(self.task_id), z3_lr_vis, self.iter)

        z1_up_vis = tensor2tb_log(z_sr[0, :, :, :])
        z2_up_vis = tensor2tb_log(z_sr[1, :, :, :])
        z3_up_vis = tensor2tb_log(z_sr[2, :, :, :])
        # self.z1_up_lst.append(z1_up_vis)
        # self.z2_up_lst.append(z2_up_vis)
        # self.z3_up_lst.append(z3_up_vis)
        self.tb_logger.add_image('Peek_{}/z1 up'.format(self.task_id), z1_up_vis, self.iter)
        self.tb_logger.add_image('Peek_{}/z2 up'.format(self.task_id), z2_up_vis, self.iter)
        self.tb_logger.add_image('Peek_{}/z3 up'.format(self.task_id), z3_up_vis, self.iter)

    # Discriminator for whole picture
    def train_d_img(self, input, is_d_input_real):
        self.optimizer_D_image.zero_grad()
        pred = self.D_image.forward(input)
        criterion = nn.L1Loss(reduction='mean')
        label = Variable(torch.ones(1).cuda(), requires_grad=False) if is_d_input_real\
            else Variable(torch.zeros(1).cuda(), requires_grad=False)
        loss = criterion(pred, label)
        return loss


    def train_g_supervised(self, g_input, d_input):
        # Zeroize gradients
        self.optimizer_G.zero_grad()

        # Generator forward pass
        out = self.G.forward(g_input)

        g_pred = out[:1, :, :, :]
        z_pred = out[1:, :, :, :]

        loss_g = nn.L1Loss(reduction='mean')(d_input, out)

        loss_g.backward(retain_graph=True)
        self.optimizer_G.step()

        self.tb_logger.add_scalar('Loss/l1'.format(self.task_id), loss_g.item(), self.iter)
        return