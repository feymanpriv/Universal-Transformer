# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import timm

from config import get_config_infer
from models import build_model
from logger import create_logger
from torchvision import transforms


MODEL_WEIGHTS = 'output/swin_large_patch4_window7_224_in22k/universal_train_swinl_win7_224_softmax/ckpt_epoch_10.pth'


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    args, unparsed = parser.parse_known_args()
    config = get_config_infer(args)

    return args, config


# Add for torchscript
def build_model_from_timm(config):
    model = timm.create_model(config.MODEL.NAME, pretrained=False, num_classes=0)
    #model.embed = torch.nn.Linear(1536, 64, bias=False)
    return model


def infer(config):
    #model = build_model_from_timm(config)
    model = build_model(config)
    
    model.eval()

    #logger.info(str(model))
    my_load_pretrained(model)
    
    x = torch.ones(1, 3, 224, 224)
    x = transforms.functional.resize(x, size=[224, 224])
    x = x / 255.0
    x = transforms.functional.normalize(x, 
                            mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    output = model(x)
    print(output)
    print(output.size())
    

def my_load_pretrained(model):

    checkpoint = torch.load(MODEL_WEIGHTS, map_location='cpu')
    state_dict = checkpoint['model']
    
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}

    model_dict.update(pretrained_dict)                                                                       
    msg = model.load_state_dict(model_dict, strict=False) 
    print(msg)

    if len(pretrained_dict) == len(state_dict):
        print('All params loaded')
    else:
        print('construct model total {} keys and pretrin model total {} keys.' \
               .format(len(model_dict), len(state_dict)))
        print('{} pretrain keys load successfully.'.format(len(pretrained_dict)))
        not_loaded_keys = [k for k in state_dict.keys() 
                                if k not in pretrained_dict.keys()]
        print(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))
        
    del checkpoint
    #torch.cuda.empty_cache()
    return state_dict
    
    

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        model = timm.create_model('swin_large_patch4_window7_224_in22k', pretrained=False, num_classes=0)
        #print(model)
        #model.eval()
        state_dict = my_load_pretrained(model)
        self.feature_extractor = model
        
        self.embed = torch.nn.Linear(1536, 64, bias=False)
        self.embed.weight.data[...] = state_dict['embed.weight']
        

    def forward(self, x):
        x = transforms.functional.resize(x, size=[224, 224])
        x = x / 255.0
        x = transforms.functional.normalize(x, 
                                    mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
        fea = self.feature_extractor(x)
        fea = self.embed(fea)
        #fea = F.normalize(fea, p=2., dim=1)
        return fea

        
    

if __name__ == '__main__':
    args, config = parse_option()

    seed = config.SEED
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    #cudnn.benchmark = True
    logger = create_logger(output_dir='./', dist_rank=0, name=f"{config.MODEL.NAME}")

    infer(config)
        
    model = MyModel()
    model.eval()
    saved_model = torch.jit.script(model)
    saved_model.save('saved_model_swin_large_win7_224_softmax_10.pt')
    
    
    
    
    