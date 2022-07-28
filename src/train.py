import os
import tqdm

from configs import Config
from data import DataGenerator
from IKENet import KernelGAN
from baseline import KernelGAN_baseline
from learner import Learner
from utils.util import create_TBlogger, setup_seed, save_model, read_image, im2tensor
from utils.peeker import Peeker
from utils import kernel_visualizer


def train(conf, tb_logger, task_id):
    if conf.INN_mode:
        gan = KernelGAN(conf, tb_logger, task_id)
    else:
        gan = KernelGAN_baseline(conf, tb_logger, task_id)

    learner = Learner(conf)
    peeker = Peeker(conf, tb_logger)
    data = DataGenerator(conf, gan)
    # g_vis, d_vis = None, None
    for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60):
        if conf.residual_supervised:
            [g_in, d_in, r_in] = data.__getitem__(iteration)
        else:
            [g_in, d_in] = data.__getitem__(iteration)

        # For Visualization Need
        # if iteration == 0:
        #     g_vis, d_vis = data.get_vis_crop(400,650)

        gan.train(g_in, d_in)

        out = gan.G.forward(g_in)
        gan.train_u(out.detach(), img=None)

        if 'cycle_consistency' in conf.other_loss:
            gan.train_ud(out.detach())

        if conf.peekHQ and (iteration+1) % conf.peek_frq == 0:
            # gan.peek()
            # peeker.peek(g_in, gan.G, gan.U, task_id)

            peeker.peek_z(gan.G, gan.U, task_id)

        learner.update(iteration, gan)
        tb_logger.flush()

    img_LR = read_image(conf.input_image_path, conf.read_mode)[:, :, :] / 255.0
    img_LR = im2tensor((img_LR))

    gan.finish()

    # peeker.peek_final(gan.G)
    if conf.save_model:
        save_model(gan, 'pretrained/')
        if conf.show_kernel:
            kernel_visualizer.show_kernel(conf, gan, 10000, tb_logger, '/home/bingbingwei/DIV2K/DIV2K_train_HR/0774.png')

def main():
    """The main function - performs kernel estimation (+ ZSSR) for all images in the 'test_images' folder"""
    import argparse
    # Parse the command line arguments
    prog = argparse.ArgumentParser()
    prog.add_argument('--input-dir', '-i', type=str, default='/home/bingbingwei/DIV2K/DIV2K_train_LR_unknown/X2/', help='path to image input directory.')
    prog.add_argument('--eval-dir', '-e', type=str, default='/home/bingbingwei/DIV2K/DIV2K_train_HR/', help='path to image input directory.')
    prog.add_argument('--output-dir', '-o', type=str, default='results', help='path to image output directory.')
    prog.add_argument('--X4', action='store_true', help='The wanted SR scale factor')
    prog.add_argument('--SR', action='store_true', help='when activated - ZSSR is not performed')
    prog.add_argument('--real', action='store_true', help='ZSSRs configuration is for real images')
    prog.add_argument('--noise_scale', type=float, default=1., help='ZSSR uses this to partially de-noise images')
    prog.add_argument('--max_iters', type=int, default=3000, help='# of iterations')
    prog.add_argument('--z_iters', type=int, default=1000, help='# of iterations')
    prog.add_argument('--optim', type=str, default='adam', choices=['adam', 'SGD'], help='Optimizer')
    prog.add_argument('--g_lr', type=float, default=2e-4, help='initial learning rate for generator')
    prog.add_argument('--gpu_id', type=str, default='0', help='gpu id number')
    prog.add_argument('--scale_factor', type=float, default=0.5, help='The downscaling scale factor')

    # Mine Parameters
    prog.add_argument('--eval', action='store_true', help='evaluation with ground truth HR image')
    prog.add_argument('--peek', action='store_true', help='Peek and evaluate kernelGAN effects')
    prog.add_argument('--peekHQ', action='store_true', help='Peek and evaluate kernelGAN effects on HQ domain image')
    prog.add_argument('--read_mode', type=str, default='RGB', choices=['RGB', 'HSV'], help='Define couple layer of INN')
    # Invertible Module Related
    prog.add_argument('--INN', action='store_true', help='Use Invertible model')
    prog.add_argument('--INN_couple', type=str, default='additive', help='Define couple layer of INN')
    prog.add_argument('--test', action='store_true', help='SR with invertible model')
    # prog.add_argument('--INN_down', type=str, default='Haar', help='Define downsample layer of INN')
    prog.add_argument('--INN_down', nargs='+', type=str, default=['Haar'],  help='Define downsample layer of INN')
    prog.add_argument('--block_num', type=int, default=4, help='Block number of invertible unit')
    prog.add_argument('--linear', action='store_true', help='Remove activation funciton')
    prog.add_argument('--residual', action='store_true', help='Use residual dense block')
    prog.add_argument('--down_mode', type=str, default='multiple_model', choices=['multiple_model', 'single_model'],
                      help='if downsample multiple times (e.x. X4 mode), choose the down method')
    prog.add_argument('--subnet_mode', type=str, default='Dense', choices=['Dense', 'ResNet'],
                             help='Choose Subnet Structure')

    # ResNet Setting
    prog.add_argument('--tri_channel', action='store_true', help='Use Tri-channel (ResNet-like) coupling layer')
    prog.add_argument('--use_res', action='store_true', help='Train ResNet Version of IKE')
    prog.add_argument('--residual_supervised', action='store_true', help='Add supervised residual constraint')
    prog.add_argument('--ressup_image_path', type=str,default='/home/bingbingwei/EVA2020/test_images/residual_supervision/X2/',
                             help='path to residual supervision image file')

    # Discriminator Related
    prog.add_argument('--mixup', action='store_true', help='Use mixup technique to improve discriminator')
    # Z upsample module related
    prog.add_argument('--z_up', type=str, default='origin', choices=['origin', 'RCAN', 'EDSR', 'ZSSR'], help='Choose Z upsample module')
    prog.add_argument('--z_freq', type=int, default=1, help='Define the update ration of z upsample module')

    # Criterion Settings
    prog.add_argument('--bicubic_loss', type=str, default='l2', choices=['l2', 'l1', 'l1+l2'], help='Choose bicubic loss')
    prog.add_argument('--other_loss', nargs='+', type=str, default=[],  help='Other constrains')

    prog.add_argument('--tb_name', type=str, default='TIME', help='Tensorboard logger directory')
    prog.add_argument('--save_model', action='store_true', help='save model Generator and Z upsampler')
    prog.add_argument('--show_kernel', action='store_true', help='Show kernel by training on HR LR pair')


    # setup_seed(3318)

    args = prog.parse_args()
    # Run the KernelGAN sequentially on all images in the input directory
    tb_logger = create_TBlogger(args)
    for id, filename in enumerate(sorted(os.listdir(os.path.abspath(args.input_dir)))):
        conf = Config().parse(create_params(filename, args))
        train(conf, tb_logger, id)
    prog.exit(0)


def create_params(filename, args):
    filename_eval = filename[:-4]+'.png'
    params = ['--input_image_path', os.path.join(args.input_dir, filename),
              '--eval_image_path', os.path.join(args.eval_dir, filename_eval),
              '--output_dir_path', os.path.abspath(args.output_dir),
              '--output_img_path', os.path.abspath(args.output_dir),
              '--ressup_image_path', args.ressup_image_path,
              '--max_iters', str(args.max_iters),
              '--z_iters', str(args.z_iters),
              '--noise_scale', str(args.noise_scale),
              '--couple_layer', args.INN_couple,
              # '--downsample', args.INN_down,
              '--bicubic_loss', args.bicubic_loss,
              '--block_num', str(args.block_num),
              '--optim', args.optim,
              '--g_lr', str(args.g_lr),
              '--filename', filename[:4],
              '--read_mode', args.read_mode,
              '--gpu_id', args.gpu_id,
              '--z_upsample', args.z_up,
              '--z_freq', str(args.z_freq),
              '--scale_factor', str(args.scale_factor),
              '--down_mode', args.down_mode,
              '--subnet_mode', args.subnet_mode]

    # default_losses = ['energy', 'tv_loss']
    default_losses = ['tv_loss']
    params.append('--other_loss')
    for arg in default_losses:
        params.append(arg)
    for arg in args.other_loss:
        params.append(arg)

    params.append('--downsample')
    for arg in args.INN_down:
        params.append(arg)

    if args.X4:
        params.append('--X4')
    if args.SR:
        params.append('--do_ZSSR')
    if args.real:
        params.append('--real_image')
    if args.INN:
        params.append('--INN_mode')
    if args.test:
        params.append('--INN_test')
    if args.eval:
        params.append('--eval')
    if args.peek:
        params.append('--peek')
    if args.peekHQ:
        params.append('--peekHQ')
    if args.mixup:
        params.append('--mixup')
    if args.linear:
        params.append('--linear')
    if args.residual:
        params.append('--residual')
    if args.use_res:
        params.append('--use_res')
    if args.tri_channel:
        params.append('--tri_channel')
    if args.residual_supervised:
        params.append('--residual_supervised')
    if args.save_model:
        params.append('--save_model')
    if args.show_kernel:
        params.append('--show_kernel')


    return params


if __name__ == '__main__':
    main()
