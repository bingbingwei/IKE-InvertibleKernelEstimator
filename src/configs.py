import argparse
import torch
import os


# noinspection PyPep8
class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.conf = None

        # Paths
        self.parser.add_argument('--img_name', default='image1', help='image name for saving purposes')
        self.parser.add_argument('--input_image_path', default=os.path.dirname(__file__) + '/training_data/input.png', help='path to LR image file')
        self.parser.add_argument('--eval_image_path', default=os.path.dirname(__file__) + '/training_data/input.png', help='path to HR image file')
        self.parser.add_argument('--ressup_image_path', default=os.path.dirname(__file__) + '/training_data/input.png', help='path to residual supervision image file')
        self.parser.add_argument('--output_dir_path', default=os.path.dirname(__file__) + '/results', help='results path')
        self.parser.add_argument('--output_img_path', default=os.path.dirname(__file__) + '/results/images', help='results path')
        self.parser.add_argument('--filename', type=str, help='image name')

        self.parser.add_argument('--np_file_path', type=str, default=os.path.dirname(__file__) + '/results/numpy', help='numpy file path')


        # Peek (check kernelGAN efficiency)
        self.parser.add_argument('--peek', action='store_true', help='Peek and evaluate kernelGAN effects')
        self.parser.add_argument('--peek_image_path', default='test_images/X4', help='path to peek(X4) image file')
        self.parser.add_argument('--peek_yaml_path', default='test_images/config', help='path to peek yaml file')
        self.parser.add_argument('--peek_frq', type=int, default=100, help='peek frequency')
        self.parser.add_argument('--peekHQ', action='store_true', help='Peek on HQ domain image')

        # Sizes
        self.parser.add_argument('--input_crop_size', type=int, default=128, help='Generators crop size')
        self.parser.add_argument('--scale_factor', type=float, default=0.25, help='The downscaling scale factor')
        self.parser.add_argument('--X4', action='store_true', help='The wanted SR scale factor')

        # Network architecture
        self.parser.add_argument('--G_chan', type=int, default=64, help='# of channels in hidden layer in the G')
        self.parser.add_argument('--D_chan', type=int, default=64, help='# of channels in hidden layer in the D')
        self.parser.add_argument('--G_kernel_size', type=int, default=13, help='The kernel size G is estimating')
        self.parser.add_argument('--D_n_layers', type=int, default=7, help='Discriminators depth')
        self.parser.add_argument('--D_kernel_size', type=int, default=7, help='Discriminators convolution kernels size')

        # Iterations
        self.parser.add_argument('--max_iters', type=int, default=3000, help='# of iterations')
        self.parser.add_argument('--z_iters', type=int, default=5000, help='# of iterations')

        # Optimization hyper-parameters
        self.parser.add_argument('--g_lr', type=float, default=2e-4, help='initial learning rate for generator')
        self.parser.add_argument('--u_lr', type=float, default=2e-4, help='initial learning rate for upsampler')
        self.parser.add_argument('--d_lr', type=float, default=2e-4, help='initial learning rate for discriminator')
        self.parser.add_argument('--beta1', type=float, default=0.25, help='Adam momentum')
        self.parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'SGD'], help='Optimizer')

        # GPU
        self.parser.add_argument('--gpu_id', type=str, default='0', help='gpu id number')

        # Kernel post processing
        self.parser.add_argument('--n_filtering', type=float, default=40, help='Filtering small values of the kernel')

        # Criterion Setting
        self.parser.add_argument('--bicubic_loss', type=str, default='l2', help='Choose bicubic loss')
        self.parser.add_argument('--other_loss', nargs='+', type=str,
                          help='Other constrains, include [image_discriminate, energy]')

        # ZSSR configuration
        self.parser.add_argument('--do_ZSSR', action='store_true', help='when activated - ZSSR is not performed')
        self.parser.add_argument('--noise_scale', type=float, default=1., help='ZSSR uses this to partially de-noise images')
        self.parser.add_argument('--real_image', action='store_true', help='ZSSRs configuration is for real images')

        # Invertible Generator Settings
        self.parser.add_argument('--eval', action='store_true', help='evaluation with ground truth HR image')
        self.parser.add_argument('--INN_mode', action='store_true', help='Use Invertible model')
        self.parser.add_argument('--INN_test', action='store_true', help='SR with invertible model')
        # self.parser.add_argument('--downsample', type=str, default='Haar', choices=['Haar','space2depth'], help='Define downsample layer of INN')
        self.parser.add_argument('--downsample', nargs='+', type=str, default=['Haar'], help='Define downsample layer of INN')
        self.parser.add_argument('--couple_layer', type=str, default='affine', choices=['affine', 'additive','none'], help='Choose from affine and additive and Haar')
        self.parser.add_argument('--block_num', type=int, default=1, help='Block number of invertible unit')
        self.parser.add_argument('--linear', action='store_true', help='Remove activation funciton')
        self.parser.add_argument('--residual', action='store_true', help='Use residual dense block')
        self.parser.add_argument('--use_res', action='store_true', help='Train ResNet Version of IKE')
        self.parser.add_argument('--tri_channel', action='store_true', help='Use Tri-channel (ResNet-like) coupling layer')
        self.parser.add_argument('--down_mode', type=str, default='multiple_model', choices=['multiple_model', 'single_model'],
                          help='if downsample multiple times (e.x. X4 mode), choose the down method')
        self.parser.add_argument('--residual_supervised', action='store_true', help='Add supervised residual constraint')
        self.parser.add_argument('--subnet_mode', type=str, default='Dense', choices=['Dense', 'ResNet'],
                          help='Choose Subnet Structure')

        # Discriminator Settings
        self.parser.add_argument('--mixup', action='store_true', help='Use mixup technique to improve discriminator')

        # Z upsample module Settings
        self.parser.add_argument('--z_upsample', type=str, default='origin', choices=['origin', 'RCAN', 'EDSR', 'ZSSR'], help='Choose Z upsample module')
        self.parser.add_argument('--z_freq', type=int, default=1, help='Define the update ration of z upsample module')

        # Others
        self.parser.add_argument('--read_mode', type=str, default='RGB', choices=['RGB', 'HSV'], help='Define couple layer of INN')
        self.parser.add_argument('--energy_k_size', type=int, default=36, help='Define Kernel Size used in Energy Loss')
        self.parser.add_argument('--save_model', action='store_true', help='save model Generator and Z upsampler')
        self.parser.add_argument('--show_kernel', action='store_true', help='Show kernel by training on HR LR pair')

    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)
        self.set_gpu_device()
        self.clean_file_name()
        if not self.conf.INN_mode:
            self.set_output_directory()

        if self.conf.INN_test:
            self.set_output_image_directory()
            self.set_numpy_file_directory()

        if self.conf.peek:
            self.set_peek_file()
        self.conf.G_structure = [7, 5, 3, 1, 1, 1]
        print("Scale Factor: %s \tZSSR: %s \tReal Image: %s" % (str(int(1/self.conf.scale_factor)), str(self.conf.do_ZSSR), str(self.conf.real_image)))
        if self.conf.INN_mode:
            print("Invertible Mode:")
            print("Coupling Layer: {} \t Downsample: {} \t Block Num: {} \t Linear: {} \t ResNet: {}"
                  .format(self.conf.couple_layer, self.conf.downsample, self.conf.block_num, self.conf.linear, self.conf.use_res))
        print("Mixup: {} \t Read Mode: {}".format(self.conf.mixup, self.conf.read_mode))
        print('Z upscale module: {}'.format(self.conf.z_upsample))
        print("Other Constrains: ", end='')
        for c in self.conf.other_loss:
            print(' {}'.format(c), end='')
        print()
        return self.conf

    def clean_file_name(self):
        """Retrieves the clean image file_name for saving purposes"""
        self.conf.img_name = self.conf.input_image_path.split('/')[-1].replace('ZSSR', '') \
            .replace('real', '').replace('__', '').split('_.')[0].split('.')[0]

    def set_gpu_device(self):
        """Sets the GPU device if one is given"""
        if os.environ.get('CUDA_VISIBLE_DEVICES', '') == '':
            os.environ['CUDA_VISIBLE_DEVICES'] = self.conf.gpu_id
            gpu_counts = len(self.conf.gpu_id.split(','))
            torch.cuda.set_device(0)
        else:
            torch.cuda.set_device(0)

    def set_output_directory(self):
        """Define the output directory name and create the folder"""
        self.conf.output_dir_path = os.path.join(self.conf.output_dir_path, self.conf.img_name)
        # In case the folder exists - stack 'l's to the folder name
        while os.path.isdir(self.conf.output_dir_path):
            self.conf.output_dir_path += 'l'
        os.makedirs(self.conf.output_dir_path)

    def set_output_image_directory(self):
        if not os.path.isdir(self.conf.output_img_path):
            os.makedirs(self.conf.output_img_path)
        self.conf.output_img_path = os.path.join(self.conf.output_img_path, self.conf.img_name)
        if not os.path.isdir(self.conf.output_img_path):
            os.makedirs(self.conf.output_img_path)

    def set_numpy_file_directory(self):
        if not os.path.isdir(self.conf.np_file_path):
            os.makedirs(self.conf.np_file_path)
        self.conf.np_file_path = os.path.join(self.conf.np_file_path, self.conf.img_name)
        if not os.path.isdir(self.conf.np_file_path):
            os.makedirs(self.conf.np_file_path)

    def set_peek_file(self):
        self.conf.peek_yaml_path = os.path.join(self.conf.peek_yaml_path, self.conf.img_name)+'.yaml'
        if os.path.isfile(self.conf.peek_yaml_path):
            return
        else:

            self.conf.peek = False
            return
