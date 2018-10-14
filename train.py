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

class Train(object):
    
    def __init__(self , args):
        self.batch_size = args.batch_size
        self.in_dim = args.input_nc
        self.out_dim = args.output_nc
        self.nz = args.nz
        self.ngf = args.ngf
        self.ndf = args.ndf
        self.nef = args.nef
        
        self.dropout = args.dropout
        
        
        self.dataset_mode = args.dataset_mode
        self.direction = args.direction
        
        
        self.num_thread = args.num_thread
        self.ckpt_dir = args.ckpt_dir
        self.sample_dir = args.sample_dir
        
        
        self.gan_mode = args.gan_mode
        self.netG = args.netG
        self.netE = args.netE
        self.netD = args.netD
        self.netD2 = args.netD2
        self.nl = args.nl
        self.where_add = args.where_add
        
        self.lr = args.lr
        self.lambda_l1 = args.lambda_L1
        self.lambda_GAN = args.lambda_GAN
        self.lambda_GAN2 = args.lambda_GAN2
        self.lambda_z = args.lambda_z
        self.lambda_kl = args.lambda_kl
        
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        
    def initialize():
        pass
    
    def forward():
        pass
    
    def update_EG():
        pass
    
    def update_D():
        pass
    
    def backpropEG():
        pass
    
    def backpropD():
        pass
    
    def update_bicycleGAN:
        pass
    
    def train():
        pass