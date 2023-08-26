'''
Author: JinYin
Date: 2022-07-01 21:46:06
LastEditors: JinYin
LastEditTime: 2023-04-24 18:49:20
FilePath: \GCTNet\tools.py
Description: 
'''
from models import *
from opts import get_opts

import librosa
import mne
import os
from scipy.fftpack import fft
from scipy import signal
from matplotlib import pyplot as plt
from audtorch.metrics.functional import pearsonr

def pick_models(opts, data_num=512, embedding=1):
    if opts.denoise_network == 'SimpleCNN':
        model = SimpleCNN(data_num).to(opts.device)
                     
    elif opts.denoise_network == 'FCNN':  
        model = FCNN(data_num).to(opts.device)
                
    elif opts.denoise_network == 'ResCNN':
        model = ResCNN(data_num).to(opts.device)
    
    elif opts.denoise_network == 'GCTNet':
        model = Generator(data_num).to(opts.device)
    
    elif opts.denoise_network == 'GeneratorCNN':
        model = GeneratorCNN(data_num).to(opts.device)
        
    elif opts.denoise_network == 'GeneratorTransformer':
        model = GeneratorTransformer(data_num).to(opts.device)
    
    elif opts.denoise_network == 'NovelCNN':
        model = NovelCNN(data_num).to(opts.device)
    
    elif opts.denoise_network == 'DuoCL':
        model = DuoCL(data_num).to(opts.device)

    elif opts.denoise_network == 'BG':
        model = BG(data_num, embedding).to(opts.device)
        
    else:
        print("model name is error!")
        pass

    return model