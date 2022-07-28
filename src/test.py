import os
import tqdm

from configs import Config
from data import DataGenerator
from IKENet import KernelGAN
from baseline import KernelGAN_baseline
from learner import Learner
from utils.util import create_TBlogger, setup_seed, load_model
from utils.peeker import Peeker

def test(conf, tb_logger, task_id):
    gan = KernelGAN(conf, tb_logger, task_id)
    load_model(gan, 'pretrained/')
    gan.test()


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
    prog.add_argument('--max_iters', type=int, default=1000, help='# of iterations')
    prog.add_argument('--optim', type=str, default='adam', choices=['adam', 'SGD'], help='Optimizer')
    prog.add_argument('--g_lr', type=float, default=2e-4, help='initial learning rate for generator')

    # Mine Parameters
    prog.add_argument('--eval', action='store_true', help='evaluation with ground truth HR image')
    prog.add_argument('--peek', action='store_true', help='Peek and evaluate kernelGAN effects')
    prog.add_argument('--peekHQ', action='store_true', help='Peek and evaluate kernelGAN effects on HQ domain image')
    prog.add_argument('--read_mode', type=str, default='RGB', choices=['RGB', 'HSV'], help='Define couple layer of INN')
    # Invertible Module Related
    prog.add_argument('--INN', action='store_true', help='Use Invertible model')
    prog.add_argument('--INN_couple', type=str, default='affine', help='Define couple layer of INN')
    prog.add_argument('--INN_test', action='store_true', help='SR with invertible model')
    prog.add_argument('--INN_down', type=str, default='Haar', help='Define downsample layer of INN')
    prog.add_argument('--block_num', type=int, default=1, help='Block number of invertible unit')
    prog.add_argument('--linear', action='store_true', help='Remove activation funciton')
    prog.add_argument('--use_res', action='store_true', help='Use Tri-channel (ResNet-like) coupling layer')


    # Discriminator Related
    prog.add_argument('--mixup', action='store_true', help='Use mixup technique to improve discriminator')
    # Z upsample module related
    prog.add_argument('--z_up', nargs='+', type=str, default=['nearest'], help='Define upsample method for z')

    # Criterion Settings
    prog.add_argument('--bicubic_loss', type=str, default='l2', choices=['l2', 'l1', 'l1+l2'], help='Choose bicubic loss')
    prog.add_argument('--other_loss', nargs='+', type=str, default=['None'], help='Other constrains, include [image_discriminate, energy, energy_all]')

    setup_seed(3318)

    args = prog.parse_args()
    # Run the KernelGAN sequentially on all images in the input directory
    tb_logger = create_TBlogger(args.INN)
    for id, filename in enumerate(os.listdir(os.path.abspath(args.input_dir))):
        conf = Config().parse(create_params(filename, args))
        test(conf, tb_logger, id)
    prog.exit(0)


def create_params(filename, args):
    filename_eval = filename[:-4]+'.png'
    params = ['--input_image_path', os.path.join(args.input_dir, filename),
              '--eval_image_path', os.path.join(args.eval_dir, filename_eval),
              '--output_dir_path', os.path.abspath(args.output_dir),
              '--max_iters', str(args.max_iters),
              '--noise_scale', str(args.noise_scale),
              '--couple_layer', args.INN_couple,
              '--downsample', args.INN_down,
              '--bicubic_loss', args.bicubic_loss,
              '--block_num', str(args.block_num),
              '--optim', args.optim,
              '--g_lr', str(args.g_lr),
              '--filename', filename[:4],
              '--read_mode', args.read_mode]
    if len(args.z_up) > 0:
        params.append('--z_upsample')
        for arg in args.z_up:
            params.append(arg)
    if len(args.other_loss) > 0:
        params.append('--other_loss')
        for arg in args.other_loss:
            params.append(arg)

    params.append('--INN_test')
    params.append('--INN_mode')

    if args.X4:
        params.append('--X4')
    if args.SR:
        params.append('--do_ZSSR')
    if args.real:
        params.append('--real_image')
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
    if args.use_res:
        params.append('--use_res')


    return params


if __name__ == '__main__':
    main()
