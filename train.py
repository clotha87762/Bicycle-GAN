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
import torcvision.utils as vutils

import os
from train import Train
from evaluate import Evaluate
import evaluate
import model

from data import CreateDataLoader
from tensorboardX import SummaryWriter

import time

class Train(object):
    
    def __init__(self , args):
        
        self.opt = args
        
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        
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
        self.A2B = True if args.direction == 'AtoB' else False
        
        
        self.num_thread = args.num_thread
        self.ckpt_dir = args.ckpt_dir
        self.sample_dir = args.sample_dir
        self.log_dir = args.log_dir
        
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
        self.sample_freq = args.sample_freq
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
    def get_scheduler(self, optimizer):
        
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - 100) / float(100 + 1)
            return lr_l
        
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        
        return scheduler
    
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
    
    def set_input(self, _input):
        
        self.realA = _input['A' if self.A2B else 'B'].to(self.device)
        self.realB = _input['B' if self.A2B else 'A'].to(self.device)
        
        assert self.realA.size[0] == self.realB.size[0]
        
        
        # 這樣動態調整會不會出問題ㄋ
        self.batch_size = self.realA.size[0]
        
        self.img_path = _input['A_path' if self.A2B else 'B_path']
    
    def set_gan_criterion(self, gan_mode = 'lsgan'):
        if gan_mode == 'lsgan':
            self.gan_criterion = nn.MSELoss()
        elif gan_mode == 'dcgan':
            self.gan_criterion = nn.BCELoss()
        else:
            raise NotImplementedError('only these tow qqq')

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
        
        self.set_gan_criterion(gan_mode = self.gan_mode)
        
        self.setup()
    
    def random_sample_z(self, batch, nz):
        return torch.randn(batch, nz).to(self.device)
    
    def encode(self, _input):
        mu, logvar = self.netE.forward(_input)
        std = logvar.mul(0.5).exp_()
        eps = self.random_sample_z(std.size(0), std.size(1))
        z = eps.mul(std).add_(mu)
        return z, mu, logvar

    
    def forward(self):
        
        half = self.batch_size // 2
        self.realA_encode = self.realA[:half]
        self.realA_random = self.realA[half:]
        self.realB_encode = self.realB[:half]
        self.realB_random = self.realB[half:]
        
        self.a_latent , self.mu , self.var = self.encode(realB_encode)
        self.fakeB_encode = self.generator(realA_encode , a_latent)
        
        self.random_z = self.random_sample_z(self.batch_size, self.nz)
        self.fakeB_random = self.generator( realA_random , random_z)
        
        self.mu2 , self.var2 = self.encoder(fakeB_random)
        
        
    
    def update_EG(self):
        # I don't know why the origin implementation set discriminator requires_grad = False QAQ
        
        self.opt_e.zero_grad()
        self.opt_g.zero_grad()
        self.backpropEG()
        
        self.opt_e.step()
        self.opt_g.step()
        
        
        self.opt_e.zero_grad()
        self.opt_g.zero_grad()
        self.backpropZ_for_Galone()
        self.opt_g.step()
    
    def update_D(self):
        
        dis_out1_real = self.discriminator1(self.realB_encode)
        dis_out1_fake = self.discriminator2(self.fakeB_encode.detach())
        
        dis_out2_real = self.discriminator2(self.realB_random)
        dis_out2_fake = self.discriminator2(self.fakeB_random.detach())
        
        real_target = torch.tensor(1.0).expand_as(dis_out1_real).to(self.device)
        fake_target = torch.tensor(0.0).expand_as(dis_out1_fake).to(self.device)
        
        gan_loss1_fake = self.gan_criterion( dis_out1_fake , fake_target)
        gan_loss1_real = self.gan_criterion( dis_out1_real , real_target)

        self.opt_d1.zero_grad() 
        
        self.d_loss1 = gan_loss1_fake + gan_loss1_real    
        self.d_loss1.backward()
        
        self.opt_d1.step()
        
        gan_loss2_fake = self.gan_criterion( dis_out2_fake , fake_target)
        gan_loss2_real = self.gan_criterion( dis_out2_real , real_target)
        
        self.opt_d2.zero_grad()
        
        self.d_loss2 = gan_loss2_fake + gan_loss2_real
        self.d_loss2.backward()
        
        self.opt_d2.step()
        
        
    
    def backpropZ_for_Galone(self):
        self.z_loss = torch.mean( torch.abs(self.mu2 - self.random_z )) * self.lambda_z
        self.z_loss.backward()
    
    def backpropEG(self):
        
        # GAN loss
        dis_out1 = self.discriminator1(self.fakeB_encode)
        dis_out2 = self.discriminator2(self.fakeB_random)
        
        real_target = torch.tensor(1.0).expand_as(dis_out1).to(self.device)
        fake_target = torch.tensor(0.0).expand_as(dis_out2).to(self.device)
        
        self.gan_loss1 = self.gan_criterion(dis_out1 , real_target)
        self.gan_loss2 = self.gan_criterion(dis_out2 , real_target)
        
        self.gan_loss = (gan_loss1 * self.lambda_GAN )   +  ( gan_loss2 * self.lambda_GAN2) 
        
        # L1 loss of images
        self.l1_loss = nn.L1Loss(fakeB_encode ,  realB_encode) * self.lambda_l1
        
        # KL loss for the vae encoder
        kl_element = self.mu.pow(2).add_(self.logvar.exp()).mul_(-1).add_(1).add_(self.logvar)
        self.kl_loss = torch.sum(kl_element).mul_(-0.5) * self.lambda_kl
        
        self.eg_loss = self.gan_loss + self.l1_loss + self.kl_loss
        self.eg_loss.backward(retain_graph = True)
        
    def backpropD(self):
        pass
    
    def update_bicycleGAN(self):
        self.forward()
        self.update_EG()
        self.update_D()
    
    def train(self):
        
        dataloader = CreateDataLoader(self.opt)
        
        writer = SummaryWriter( log_dir = self.log_dir)
        
        for i in range(self.epoch):
            
            epoch_start = time.time()
            epoch_step = 0
            
            for j , data in enumerate(dataloader):
                
                self.set_input(data)
                self.update_bicycleGAN()
                
                self.step = self.step + 1
                
                epoch_step += data['A'].size[0]
                
                if i % self.save_freq == 0:
                    torch.save(
                            { 'generator' : self.generator.state_dict(),
                              'discriminator1' : self.discriminator1.state_dict(),
                              'discriminator2' : self.discriminator2.state_dict(),
                              'encoder' : self.encoder.state_dict(),
                              'opt_g' : self.opt_g,
                              'opt_d1' : self.opt_d1,
                              'opt_d2' : self.opt_d2,
                              'opt_e' : self.opt_e,
                              'step' : self.step} , os.path.join(self.ckpt_dir,self.run_name+'.ckpt'))
                
                if self.step % self.print_freq == 0 :
                    print ("[Epoch %d/%d] [Batch %d/%d] [D loss1: %f] [D loss2: %f] [Z loss: %f] [G loss: %f]" % \
                       (i, self.epoch  , j, len(dataloader), self.d_loss1.item(), self.d_loss2.item() , self.z_loss.item(), self.eg_loss.item() ))
                
                if self.step % 200 == 0 :
                    writer.add_scalar('loss/g1_loss' , self.gan_loss1 , self.step)
                    writer.add_scalar('loss/g2_loss' , self.gan_loss2 , self.step)
                    writer.add_scalar('loss/d1_loss' , self.d_loss1 , self.step)
                    writer.add_scalar('loss/d2_loss' , self.d_loss2 , self.step)
                    writer.add_scalar('loss/z_loss' , self.z_loss , self.step)
                    writer.add_scalar('loss/l1_loss' , self.l1_loss , self.step)
                    writer.add_scalar('loss/kl_loss' , self.kl_loss , self.step)
                
                if self.step % self.sample_freq == 0 :
                    encode_pair = torch.cat( [self.realA_encode , self.realB_encode , self.fakeB_encode ] , dim = 0 )
                    vutils.save_image( encode_pair , sample_path + '/%d.png' % step , normalize=True)
                
                # write the logs
            #############################################
            epoch_end = time.time()
            self.update_learning_rate()
            
            if i % self.save_freq == 0:
                    torch.save({ 'generator' : self.generator.state_dict(),
                              'discriminator1' : self.discriminator1.state_dict(),
                              'discriminator2' : self.discriminator2.state_dict(),
                              'encoder' : self.encoder.state_dict(),
                              'opt_g' : self.opt_g,
                              'opt_d1' : self.opt_d1,
                              'opt_d2' : self.opt_d2,
                              'opt_e' : self.opt_e,
                              'step' : self.step} , os.path.join(self.ckpt_dir,self.run_name+'.ckpt'))
            
            
            
        
        
        
        
        
        
        
        
        