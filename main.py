# -*- coding: utf-8 -*-

import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.autograd as autograd

import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torcvision.utils as utils

import os
from train import Train
from evaluate import Evaluate
import evaluate

from tensorboardX import SummaryWriter

# all codes in the 'data' folder is borrow from the original bicycle gan github

parser = argparse.ArgumentParser()

parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--nz', type=int, default=8, help='#latent vector')
parser.add_argument('--nef', type=int, default=64, help='# of encoder filters in first conv layer')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')
parser.add_argument('--run_name', type=str, default='train', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='not implemented')
parser.add_argument('--dataset_mode', type=str, default='aligned', help='aligned,single')
parser.add_argument('--model', type=str, default='bicycle_gan', help='chooses which model to use. bicycle,, ...')
parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
parser.add_argument('--epoch', type = int , default= 200 , help = 'how many epoch?')
parser.add_argument('--num_threads', default=4, type=int, help='# sthreads for loading data')
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints', help='models are saved here')
parser.add_argument('--log_dir', type=str, default='./logs', help='models are saved here')
parser.add_argument('--sample_dir', type=str, default='./sample_dir', help='models are saved here')


parser.add_argument('--shuffle', action='store_false', help='if false, read the data serially')
parser.add_argument('--dropout', action='store_true', help='use dropout for the generator')
parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')

# models
parser.add_argument('--num_Ds', type=int, default=2, help='number of Discrminators')
parser.add_argument('--gan_mode', type=str, default='lsgan', help='dcgan|lsgan')
parser.add_argument('--netD', type=str, default='basic_256_multi', help='selects model to use for netD')
parser.add_argument('--netD2', type=str, default='basic_256_multi', help='selects model to use for netD')
parser.add_argument('--netG', type=str, default='unet_256', help='selects model to use for netG')
parser.add_argument('--netE', type=str, default='resnet_256', help='selects model to use for netE')
parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
parser.add_argument('--upsample', type=str, default='basic', help='basic | bilinear')
parser.add_argument('--nl', type=str, default='relu', help='non-linearity activation: relu | lrelu | elu')

# extra parameters
parser.add_argument('--where_add', type=str, default='all', help='input|all|middle; where to add z in the network G')
parser.add_argument('--conditional_D', action='store_true', help='if use conditional GAN for D')
parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
parser.add_argument('--center_crop', action='store_true', help='if apply for center cropping for the test')
parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')

# train options
parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
parser.add_argument('--display_port', type=int, default=8097, help='visdom display port')
parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
parser.add_argument('--update_html_freq', type=int, default=4000, help='frequency of saving training results to html')
parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
parser.add_argument('--sample_freq', type=int, default=200, help='frequency of saving the latest results')
parser.add_argument('--save_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
parser.add_argument('--train_over', action='store_true', help='continue training: load the latest model')
#parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')

parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')

parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
#parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')


# learning rate
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
parser.add_argument('--lr_decay_iters', type=int, default=100, help='multiply by a gamma every lr_decay_iters iterations')

# lambda parameters
parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for |B-G(A, E(B))|')
parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight on D loss. D(G(A, E(B)))')
parser.add_argument('--lambda_GAN2', type=float, default=1.0, help='weight on D2 loss, D(G(A, random_z))')
parser.add_argument('--lambda_z', type=float, default=0.5, help='weight for ||E(G(random_z)) - random_z||')
parser.add_argument('--lambda_kl', type=float, default=0.01, help='weight for KL loss')
parser.add_argument('--use_same_D', action='store_true', help='if two Ds share the weights or not')


# eval options
parser.add_argument('--results_dir', type=str, default='../results/', help='saves results here.')
parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
parser.add_argument('--n_samples', type=int, default=5, help='#samples')
parser.add_argument('--no_encode', action='store_true', help='do not produce encoded image')
parser.add_argument('--sync', action='store_true', help='use the same latent code for different input images')


parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio for the results')

parser.add_argument('--phase' , type=str , defualt = 'train' , help = 'train/eval')

args = parser.parse_args()

if __name__ == '__main__' :
    
    if args.phase == 'train' :
        trainer = Train(args)
        trainer.train()
    elif args.phase == 'eval' :
        evaluater = Evaluate(args)
        evaluater.evaluate()


    