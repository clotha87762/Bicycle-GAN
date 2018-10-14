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



class SingleImageLoader(utils.data.Dataset):
    
    def __init__(self , args ):
        super( SingleImageLoader , self).__init__()
        self.args = args
        
     def name(self):
        return 'BaseDataset'

    def initialize(self):
        