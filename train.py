# -*- coding: utf-8 -*-
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.autograd as autograd
from torch.optim import lr_scheduler

import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torcvision.utils as utils

import os
from train import Train
from evaluate import Evaluate
import evaluate
import model

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
        
        self.norm = args.norm
        self.dropout = args.dropout
        self.upsample = args.upsample
        self.nl = args.nl
        
        self.imsize = args.fineSize
        
        self.dataset_mode = args.dataset_mode
        self.direction = args.direction
        
        
        self.num_thread = args.num_thread
        self.ckpt_dir = args.ckpt_dir
        self.sample_dir = args.sample_dir
        
        self.num_d = args.num_Ds
        self.gan_mode = args.gan_mode
        self.netG = args.netG
        self.netE = args.netE
        self.netD = args.netD
        self.netD2 = args.netD2
        self.where_add = args.where_add
        
        self.lr = args.lr
        self.beta1 = args.beta1
        self.lambda_l1 = args.lambda_L1
        self.lambda_GAN = args.lambda_GAN
        self.lambda_GAN2 = args.lambda_GAN2
        self.lambda_z = args.lambda_z
        self.lambda_kl = args.lambda_kl
        
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.train_over = args.train_over
        
        self.verbose = args.verbose
        
        self.gpu_ids = args.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        
        self.run_name =  args.run_names
        self.step = 0
    
        
        self.initialize()
        
    def setup(self):
        
        self.schedulers = [ self.get_scheduler(optimizer) for optimizer in self.optimizers]

        # load models
        if not self.train_over:
            ckpt_path = os.path.join(self.ckpt_dir,self.run_name)
            if os.path.isfile(os.path.join(ckpt_path,args.run_name+'.ckpt')):
                print('found ckpt file' + os.path.join(ckpt_path,args.run_name+'.ckpt'))
                ckpt = torch.load(os.path.join(ckpt_path,args.run_name+'.ckpt'))
                self.generator.load_state_dict(ckpt['generator'])
                self.discriminator1.load_state_dict(ckpt['discriminator1'])
                self.discriminator2.load_state_dict(ckpt['discriminator2'])
                self.encoder.load_state_dict(ckpt['encoder'])
                
                self.opt_d1.load_state_dict(ckpt['opt_d1'])
                self.opt_d2.load_state_dict(ckpt['opt_d2'])
                self.opt_g.load_state_dict(ckpt['opt_g'])
                self.opt_e.load_state_dict(ckpt['opt_e'])
                self.step = ckpt['step']
            


    # code borrow from https://github.com/junyanz/BicycleGAN/blob/master/models/networks.py
    def get_scheduler(optimizer):
        
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - 100) / float(100 + 1)
            return lr_l
        
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        
        return scheduler


    def initialize(self):
        
        if self.imsize == 256:
            num_down = 8
            num_layer = 3
            blocks = 4
        elif self.imsize == 128:
            num_down = 7
            num_layer = 2
            blocks = 3
        else:
            raise NotImplementedError('not support other size')
        
        if self.where_add == 'input' :
            self.generator = model.Generator_Unet_in( self.in_dim,self.out_dim, self.nz, self.ngf ,self.dropout, 
                                                     norm = self.norm, nl = self.nl , upsample = self.upsample , num_down = num_down)
        elif self.where_add == 'all' :
            self.generator = model.Generator_Unet_all(self.in_dim,self.out_dim, self.nz, self.ngf ,self.dropout, 
                                                     norm = self.norm, nl = self.nl , upsample = self.upsample , num_down = num_down )
        
        self.discriminator1 = model.Discriminator( self.in_dim , self.imsize , self.nz , self.ndf , dropout = self.dropout , self.num_d = 2, 
                                                  norm = self.norm , gan_mode = self.gan_mode , num_layer = num_layer)
        self.discriminator2 = model.Discriminator( self.in_dim , self.imsize , self.nz , self.ndf , dropout = self.dropout , self.num_d = 2, 
                                                  norm = self.norm , gan_mode = self.gan_mode , num_layer = num_layer)
        
        self.encoder = model.Encoder_resnet( ndf = self.nef , blocks = blocks ,in_dim = self.in_dim , 
                                            out_dim = self.out_dim , norm = self.norm  , nl =  self.nl)
        
        
        self.generator = model.init_net(self.generator,gpu_ids = self.gpu_ids)
        self.discriminator1 = model.init_net(self.discriminator1,gpu_ids = self.gpu_ids)
        self.discriminator2 = model.init_net(self.discriminator2, gpu_ids = self.gpu_ids)
        self.encoder = model.init_net(self.encoder ,gpu_ids = self.gpu_ids)
        
        self.optimizers = []
        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr = self.lr , betas = (self.beta1,0.999))
        self.opt_e = torch.optim.Adam(self.encoder.parameters(), lr = self.lr , betas = (self.beta1,0.999))
        self.opt_d1 = torch.optim.Adam(self.discriminator1.parameters(), lr = self.lr , betas = (self.beta1,0.999))
        self.opt_d2 = torch.optim.Adam(self.discriminator2.parameters(), lr = self.lr , betas = (self.beta1,0.999))
        
        self.optimizers.append(self.opt_g)
        self.optimizers.append(self.opt_d1)
        self.optimizers.append(self.opt_d2)
        self.optimizers.append(self.opt_e)
        
        
        self.setup()
    
    def random_sample_z(batch, nz):
        return torch.randn(batch, nz)
    
    def forward(self):
        pass
    
    def update_EG(self):
        pass
    
    def update_D(self):
        pass
    
    def backpropEG(self):
        pass
    
    def backpropD(self):
        pass
    
    def update_bicycleGAN(self):
        pass
    
    def train(self):
        pass