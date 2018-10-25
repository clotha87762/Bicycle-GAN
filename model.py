# -*- coding: utf-8 -*-

import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.autograd as autograd

import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.nn.functional as F
import functools
from torch.nn import init

import os
from tensorboardX import SummaryWriter

# code of init_weights & init_net are borrow from https://github.com/junyanz/BicycleGAN/blob/master/models/networks.py

def init_weights(net, init_type='normal', gain=0.02):
    
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='xavier', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        #net.to(gpu_ids[0])
        net = net.cuda()
        gpu_list = [ int(i) for i in gpu_ids.split(',')]
        if len(gpu_list) > 1 :
            net = torch.nn.DataParallel(net, gpu_list)
    init_weights(net, init_type)
    return net



def Upsample( in_dim , out_dim , upsample = 'basic'):
    
    if upsample == 'basic':
        layers = nn.Sequential(
                nn.ConvTranspose2d(in_dim,out_dim,kernel_size = 4 , stride = 2 , padding = 1)
                )
    elif upsample == 'bilinear':
        layers = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_dim,out_dim,kernel_size = 3 ,stride = 1 , padding = 0)
                )
    else:
        raise NotImplementedError('only these two...')
        
    return layers

def Norm_Layer(norm , channel ):
    if norm == 'batch':
        layer =  functools.partial(nn.BatchNorm2d , affine = True , num_features = channel)
    elif norm == 'instance':
        layer  = functools.partial(nn.InstanceNorm2d, affine = False , num_features = channel )
    elif norm == 'group' :
        layer =  functools.partial(nn.GroupNorm , num_groups = (channel / 10) , num_channels = channel )
    elif norm == 'none' :
        layer = None
    else:
        raise NotImplementedError('only these two qwq')
    return layer

def NL(nl):
    if nl == 'relu':
        return nn.ReLU(True)
    elif nl == 'lrelu':
        return nn.LeakyReLU(0.2,True)
    elif nl == 'elu':
        return nn.ELU(inplace = True)
    else:
        raise NotImplementedError('only these three qwq')


class Generator_Unet_in(nn.Module):
    
    def __init__(self , in_dim , out_dim , nz , ngf ,dropout = False , norm = 'instance', nl = 'relu' , upsample = 'basic', num_down = 8 ):
        super(Generator_Unet_in , self).__init__()
        encs = []
        decs = []
        self.ngf = ngf
        self.nz = nz
        self.in_dim = in_dim
        self.out_dim= out_dim
        self.num_down = num_down
        # encoder
        
        enc = nn.Conv2d(in_dim+nz, ngf ,4,2,1)
        encs.append(enc)
        enc = nn.Sequential(
                NL('lrelu'),
                nn.Conv2d(ngf , ngf*2 ,4 , 2 ,1),
                Norm_Layer( norm , ngf * 2)()
                )
        encs.append(enc)
        enc = nn.Sequential(
                NL('lrelu'),
                nn.Conv2d(ngf*2 , ngf * 4 ,4 , 2, 1),
                Norm_Layer( norm , ngf * 4)()
                )
        encs.append(enc)
        enc = nn.Sequential(
                NL('lrelu'),
                nn.Conv2d(ngf*4 , ngf * 8, 4 , 2, 1),
                Norm_Layer( norm , ngf * 8)()
                )
        encs.append(enc)

        
        for i in range(num_down - 5):
            enc = nn.Sequential(
                NL('lrelu'),
                nn.Conv2d(ngf*8 , ngf * 8,4  , 2, 1),
                Norm_Layer( norm , ngf * 8)()
                )
            encs.append(enc)
            
        enc = nn.Sequential(
            NL('lrelu'),
            nn.Conv2d(ngf*8 , ngf * 8 , 4 , 2, 1),
            )
        encs.append(enc)
        
        # decoder
        
        dec = nn.Sequential(
                NL(nl),
                Upsample(ngf*8,ngf*8,upsample = upsample),
                Norm_Layer(norm , ngf*8)()
                )
        decs.append(dec)
        
        for i in range(num_down - 5):
            dec = []
            dec.append(NL(nl))
            dec.append( Upsample(ngf * 8 * 2 , ngf * 8 , upsample = upsample))
            dec.append(Norm_Layer(norm, ngf * 8 * 2)())
            if dropout:
                dec.append(nn.Dropout(0.5))
            decs.append( nn.Sequential(*dec) )
        
        dec = nn.Sequential(
                NL(nl),
                Upsample(ngf * 8 * 2, ngf * 4 , upsample = upsample),
                Norm_Layer(norm , ngf * 4 )()
                )
        decs.append(dec)
        dec = nn.Sequential(
                NL(nl),
                Upsample(ngf * 4 * 2, ngf * 2, upsample = upsample),
                Norm_Layer(norm , ngf * 2 )()
                )
        decs.append(dec)
        dec = nn.Sequential(
                NL(nl),
                Upsample(ngf *2 * 2, ngf * 1 , upsample = upsample),
                Norm_Layer(norm , ngf  )()
                )
        decs.append(dec)
        dec = nn.Sequential(
                NL(nl),
                Upsample(ngf * 1 * 2, out_dim , upsample = upsample),
                nn.Tanh()
                )
        decs.append(dec)
        self.encs = nn.ModuleList(encs)
        self.decs = nn.ModuleList(decs)
        

        
    def forward(self , _input , z = None):
        
        enc_outs = []
        
        if self.nz <= 0 :  
            out = _input    
        else :
            z = z.view(z.size(0) , z.size(1) , 1 , 1).expand(z.size(0) ,z.size(1) , _input.size(2) , _input.size(3))
            out = torch.cat( [_input , z] , dim = 1 )
        
        
        for enc in self.encs:
            out = enc(out)
            enc_outs.append(out)
            
        for i , dec in enumerate(self.decs):
            if i == 0:
                out = dec(out)
            else:
                out = torch.cat([out, enc_outs[self.num_down - i - 1]] , dim = 1)
                out = dec(out)
                    
        return out
    


class Generator_Unet_all(nn.Module):
    
    def __init__(self , in_dim , out_dim , nz , ngf ,dropout = True , norm = 'instance', nl = 'lrelu' , upsample = 'basic', num_down = 8 ):
        
        super(Generator_Unet_all , self).__init__()
        
        encs = []
        decs = []
        self.ngf = ngf
        self.nz = nz
        self.in_dim = in_dim
        self.out_dim= out_dim
        self.num_down = num_down
        # encoder
        
        enc = nn.Conv2d(in_dim+nz, ngf  ,4,2,1)
        encs.append(enc)
        enc = nn.Sequential(
                NL('lrelu'),
                nn.Conv2d(ngf + nz , ngf*2  ,4 , 2 ,1),
                Norm_Layer( norm , ngf * 2)()
                )
        encs.append(enc)
        enc = nn.Sequential(
                NL('lrelu'),
                nn.Conv2d(ngf*2 + nz, ngf * 4 , 4  , 2, 1),
                Norm_Layer( norm , ngf * 4)()
                )
        encs.append(enc)
        enc = nn.Sequential(
                NL('lrelu'),
                nn.Conv2d(ngf*4 + nz , ngf * 8 ,4 , 2, 1),
                Norm_Layer( norm , ngf * 8)()
                )
        encs.append(enc)

        
        for i in range(num_down - 5):
            enc = nn.Sequential(
                NL('lrelu'),
                nn.Conv2d(ngf*8 + nz, ngf * 8,4 , 2, 1),
                Norm_Layer( norm , ngf * 8)()
                )
            encs.append(enc)
            
        enc = nn.Sequential(
            NL('lrelu'),
            nn.Conv2d(ngf*8 + nz , ngf * 8 , 4 , 2, 1),
            )
        encs.append(enc)
        
        # decoder
        
        dec = nn.Sequential(
                NL(nl),
                Upsample(ngf*8,ngf*8,upsample = upsample),
                Norm_Layer(norm , ngf*8)()
                )
        decs.append(dec)
        
        for i in range(num_down - 5):
            dec = []
            dec.append(NL(nl))
            dec.append( Upsample(ngf * 8 * 2 , ngf * 8 , upsample = upsample))
            dec.append(Norm_Layer(norm, ngf * 8 )())
            if dropout:
                dec.append(nn.Dropout(0.5))
            decs.append( nn.Sequential(*dec) )
        
        dec = nn.Sequential(
                NL(nl),
                Upsample(ngf * 8 * 2, ngf * 4 , upsample = upsample),
                Norm_Layer(norm , ngf * 4 )()
                )
        decs.append(dec)
        dec = nn.Sequential(
                NL(nl),
                Upsample(ngf * 4 * 2, ngf * 2 , upsample = upsample),
                Norm_Layer(norm , ngf * 2)()
                )
        decs.append(dec)
        dec = nn.Sequential(
                NL(nl),
                Upsample(ngf * 2 * 2, ngf * 1 , upsample = upsample),
                Norm_Layer(norm , ngf * 1 )()
                )
        decs.append(dec)
        dec = nn.Sequential(
                NL(nl),
                Upsample(ngf * 1 * 2, out_dim , upsample = upsample),
                nn.Tanh()
                )
        decs.append(dec)
        self.encs = nn.ModuleList(encs)
        self.decs = nn.ModuleList(decs)
        

        
    def forward(self , _input , z = None):
        
        enc_outs = []
        
        #encode 
        
        if self.nz <= 0 :  
            out = _input
            for enc in self.encs:
                out = enc(out)
                enc_outs.append(out)
        
        else :
            
            
            out = _input
           
            
            for enc in self.encs:
                temp = z.view(z.size(0) , z.size(1) , 1 , 1).expand(z.size(0) ,z.size(1), out.size(2), out.size(3))
                
                out = torch.cat( [out , temp] , dim = 1 )
                
                out = enc(out)
                enc_outs.append(out)
           
        # decode
        for i , dec in enumerate(self.decs):
            if i == 0:
                out = dec(out)
            else:
                out = torch.cat([out, enc_outs[self.num_down - i - 1]] , dim = 1)
                out = dec(out)
                #print(out.size())
                    
        return out

'''
class GANLoss(nn.Module):
    
    def __init__(self , real_target , fake_target , gan_mode = 'lsgan' ):
        if gan_mode == 'lsgan' :
            self.loss = nn.MSELoss()
        elif gan_mode == 'dcgan' :
            self.loss = nn.BCELoss()
        else:
            raise NotImplementedError('no such gan!!')
'''


class Encoder_resnet(nn.Module):
    
    def __init__(self, ndf , nz , blocks = 4 ,in_dim = 3 , out_dim = 3 , norm = 'instance' , nl = 'relu' ):
        
        super( Encoder_resnet , self).__init__()
        
        max_ndf = 4
        self.init_conv =  nn.Conv2d(in_dim , ndf, kernel_size=4, stride=2, padding=1, bias=True)
        self.res_blocks = nn.ModuleList()
        
        for i in range(1 , blocks):
            feature = min( ndf * (i) , ndf * max_ndf)
            next_feature = min( ndf * (i+1) , ndf * max_ndf)
            self.res_blocks.append(ResBlock(feature,next_feature, norm , nl))
        
        self.avg_pool = nn.AvgPool2d(8)
        self.nl_layer = NL(nl)
        self.fc = nn.Linear(ndf * max_ndf , nz)
        self.fc_var = nn.Linear( ndf * max_ndf , nz)
        
    def forward(self,x):
        
        out = self.init_conv(x)
        for block in self.res_blocks:
            out = block(out)
        out = self.nl_layer(out)
        out = self.avg_pool(out)
        #print(out.size())
        
        out = out.view( out.size(0) , -1 )
        
        
        
        mu = self.fc(out)
        var = self.fc_var(out)
        
        return mu , var
        

        
class ResBlock(nn.Module):
    
    def __init__(self, in_dim , out_dim, norm,  nl ):
        super(ResBlock , self).__init__()
        sequence = [Norm_Layer(norm,in_dim)()]
        sequence += [ NL(nl),
                      nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias= True),
                      Norm_Layer(norm,out_dim)(),
                      NL(nl),
                      nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias= True),
                      nn.AvgPool2d(2)
                     ]
        
        identity = [nn.AvgPool2d(kernel_size = 2, stride = 2), 
                    nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=1,padding =0)]
        
        self.mainblock = nn.Sequential(*sequence)
        self.idblock = nn.Sequential(*identity)
    
    def forward(self , x):
        out = self.mainblock(x) + self.idblock(x)
        return out
        

class Encoder_conv(nn.Module):
    
    def __init__(self):
        raise NotImplementedError('not implemented yet')
    
class Discriminator(nn.Module):
    
    # 3 layer for 256 size input, 2 for 128 size input
    def __init__(self , in_dim , imsize , nz , ndf ,dropout = False , num_d = 2, norm = 'instance' , gan_mode = 'lsgan' , num_layer = 3  ):
        
        super(Discriminator , self).__init__()
        
        self.imsize = imsize
        self.nz = nz
        self.ndf = ndf
        self.in_dim = in_dim
        self.num_d = num_d
        
        self.down_layer = nn.AvgPool2d(3 , stride = 2 , padding = [1,1] ,count_include_pad = False)
        #self.models = nn.ModuleList()
        
        models_list = []
        
        
        temp = ndf 
        for i in range(num_d):
            dis = self.create_discriminator( in_dim , temp ,num_layer , norm = norm , gan_mode = gan_mode )
            temp = int((temp/2))
            models_list.append( nn.Sequential(*dis) )
        
        self.models = nn.ModuleList( models_list )
    
    
    def forward(self, _input ):
        output = []
        x = _input
        for i in range(self.num_d):
            output.append( self.models[i](x))
            if i != self.num_d - 1 :
                x = self.down_layer(x)
        return output
        
    def create_discriminator( self , in_dim , ndf  , num_layer , norm = 'instance' , gan_mode = 'lsgan'):
        
        sequence = [nn.Conv2d( in_dim , ndf, kernel_size=4, stride=2, padding=1), 
                    nn.LeakyReLU(0.2, True)]
        
        for i in range(0,num_layer-1):
            mul = min(2**(i) , 8)
            next_mul = min(2**(i+1) , 8 )
            feature_count = ndf * mul
            next_feature_count = ndf * next_mul
            
            sequence += [
                nn.Conv2d(feature_count, next_feature_count ,
                          kernel_size=4, stride=2, padding=1),
                Norm_Layer(norm , next_feature_count)() , 
                nn.LeakyReLU(0.2, True)
            ]
        
        
        mul = min(2**(num_layer) , 8)
        
        sequence += [
                nn.Conv2d( next_feature_count, ndf * mul ,
                          kernel_size=4, stride=1, padding=1),
                Norm_Layer(norm , ndf * mul)() ,
                nn.LeakyReLU(0.2, True)
            ]
        
        sequence += [nn.Conv2d( ndf * mul , 1, kernel_size=4 , stride=1, padding=1)]
        
        if gan_mode == 'dcgan':
            sequence += [ nn.Sigmoid() ]
        
        return sequence
        
    