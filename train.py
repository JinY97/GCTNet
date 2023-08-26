'''
Author: Yin Jin
Date: 2022-03-08 19:50:50
LastEditTime: 2023-04-24 19:02:10
LastEditors: JinYin
'''

import torch.nn.functional as F
import argparse, torch
from matplotlib.font_manager import weight_dict
import torch.optim as optim
import numpy as np
from tqdm import trange
from opts import get_opts
from audtorch.metrics.functional import pearsonr

import os
from models import *
from loss import denoise_loss_mse
from torch.utils.tensorboard import SummaryWriter
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

from preprocess.DenoisenetPreprocess import *
from tools import pick_models

from torchsummary import summary as summary

loss_type = "feature+cls" 
if loss_type == "feature":
    w_c = 0
    w_f = 0.05      # 0, 0.01, 0.05, 0.1, 0.5
elif loss_type == "cls":
    w_c = 0.05      # 0, 0.001, 0.005, 0.01, 0.05
    w_f = 0
elif loss_type == "feature+cls":
    w_f = 0.05
    w_c = 0.05

def cal_SNR(predict, truth):
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()

    PS = np.sum(np.square(truth), axis=-1)  # power of signal
    PN = np.sum(np.square((predict - truth)), axis=-1)  # power of noise
    ratio = PS / PN
    return torch.from_numpy(10 * np.log10(ratio))

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
               
def train(opts, model, train_log_dir, val_log_dir, data_save_path, fold):
    if opts.noise_type == 'ECG':
        EEG_train_data, NOS_train_data, EEG_val_data, NOS_val_data, EEG_test_data, NOS_test_data = load_data_ECG(opts.EEG_path, opts.NOS_path, fold)
    elif opts.noise_type == 'EOG':
        EEG_train_data, NOS_train_data, EEG_val_data, NOS_val_data, EEG_test_data, NOS_test_data = load_data(opts.EEG_path, opts.NOS_path, fold)
    elif opts.noise_type == 'EMG':
        EEG_train_data, NOS_train_data, EEG_val_data, NOS_val_data, EEG_test_data, NOS_test_data = load_data(opts.EEG_path, opts.NOS_path, fold)
    elif opts.noise_type == 'Hybrid':
        EMG_path = "./data/EMG_shuffle.npy"
        EOG_path = "./data/EOG_shuffle.npy"
        EEG_train_data, NOS_train_data, EEG_val_data, NOS_val_data, EEG_test_data, NOS_test_data = load_data_hybrid(opts.EEG_path, EMG_path, EOG_path, fold)
    train_data = EEGwithNoise(EEG_train_data, NOS_train_data, opts.batch_size)
    val_data = EEGwithNoise(EEG_val_data, NOS_val_data, opts.batch_size)
    test_data = EEGwithNoise(EEG_test_data, NOS_test_data, opts.batch_size)
    
    model_d = Discriminator().to('cuda:5')
    
    model_d.apply(weights_init)
    model.apply(weights_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.9), eps=1e-8)
    optimizer_D = torch.optim.Adam(model_d.parameters(), lr=0.0001)
        
    best_mse = 100
    if opts.save_result:
        train_summary_writer = SummaryWriter(train_log_dir)
        val_summary_writer = SummaryWriter(val_log_dir)
        f = open(data_save_path + "result.txt", "a+")
    
    for epoch in range(opts.epochs):
        model.train()
        model_d.train()
        losses = []
        for batch_id in trange(train_data.len()):
            x_t, y_t = train_data.get_batch(batch_id)
            x_t, y_t = torch.Tensor(x_t).to(opts.device).unsqueeze(dim=1), torch.Tensor(y_t).to(opts.device)
            
            y_original = y_t
            if batch_id % 1 == 0:
                p_t = model(x_t).view(x_t.shape[0], -1)
                fake_y, _, _, _ = model_d(p_t.unsqueeze(dim=1))
                real_y, _, _, _ = model_d(y_t.unsqueeze(dim=1))
                
                d_loss = 0.5 * (torch.mean((fake_y) ** 2)) + 0.5 * (torch.mean((real_y - 1) ** 2))
                
                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()
            
            if batch_id % 1 == 0:
                p_t = model(x_t).view(x_t.shape[0], -1)

                fake_y, _, fake_feature2, _ = model_d(p_t.unsqueeze(dim=1))
                _, _, true_feature2, _ = model_d(y_t.unsqueeze(dim=1))

                y_t = y_original
                loss = denoise_loss_mse(p_t, y_t)
                
                if loss_type == "cls":
                    g_loss = loss + w_c * (torch.mean((fake_y - 1) ** 2)) 
                elif loss_type == "feature": 
                    g_loss = loss + w_f * denoise_loss_mse(fake_feature2, true_feature2)
                elif loss_type == "feature+cls": 
                    g_loss = loss + w_f * denoise_loss_mse(fake_feature2, true_feature2) + w_c * (torch.mean((fake_y - 1) ** 2))  

                optimizer_D.zero_grad()
                optimizer.zero_grad()
                g_loss.backward()
                optimizer.step()
                    
                losses.append(g_loss.detach())
                
        train_data.shuffle()
        train_loss = torch.stack(losses).mean().item()

        if opts.save_result:
            train_summary_writer.add_scalar("Train loss", train_loss, epoch)
        
        # val
        model.eval()
        losses = []
        for batch_id in range(val_data.len()):
            x_t, y_t = val_data.get_batch(batch_id)
            x_t, y_t = torch.Tensor(x_t).to(opts.device).unsqueeze(dim=1), torch.Tensor(y_t).to(opts.device)
            
            with torch.no_grad():
                p_t = model(x_t).view(x_t.shape[0], -1)
                loss = ((p_t - y_t) ** 2).mean(dim=-1).sqrt().detach()
                losses.append(loss)
        val_mse = torch.cat(losses, dim=0).mean().item()
        val_summary_writer.add_scalar("Val loss", val_mse, epoch)
        
        # test
        model.eval()
        losses = []
        single_acc, single_snr = [], []
        clean_data, output_data, input_data = [], [], []
        correct_d, sum_d = 0, 0
        for batch_id in range(test_data.len()):
            x_t, y_t = test_data.get_batch(batch_id)
            x_t, y_t = torch.Tensor(x_t).to(opts.device).unsqueeze(dim=1), torch.Tensor(y_t).to(opts.device)

            with torch.no_grad():
                p_t = model(x_t).view(x_t.shape[0], -1)
                loss = (((p_t - y_t) ** 2).mean(dim=-1).sqrt() / (y_t ** 2).mean(dim=-1).sqrt()).detach()
                losses.append(loss.detach())
                single_acc.append(pearsonr(p_t, y_t))
                single_snr.append(cal_SNR(p_t, y_t))
                
                p_t = model(x_t).view(x_t.shape[0], -1)
                
                fake_y, _, _, _ = model_d(p_t.unsqueeze(dim=1))
                real_y, _, _, _ = model_d(y_t.unsqueeze(dim=1))
                
                correct_d += torch.sum(torch.where(fake_y < 0.5, 1, 0)) + torch.sum(torch.where(real_y > 0.5, 1, 0))
                sum_d += p_t.shape[0] * 2
                    
            output_data.append(p_t.cpu().numpy()), clean_data.append(y_t.cpu().numpy()), input_data.append(x_t.cpu().numpy())
                    
        test_rrmse = torch.cat(losses, dim=0).mean().item()
        sum_acc = torch.cat(single_acc, dim=0).mean().item()
        sum_snr = torch.cat(single_snr, dim=0).mean().item()
        
        val_summary_writer.add_scalar("test rrmse", test_rrmse, epoch)
        
        # save best result
        if val_mse < best_mse:
            best_mse = val_mse
            best_acc = sum_acc
            best_snr = sum_snr
            best_rrmse = test_rrmse
            print("Save best result")
            f.write("Save best result \n")
            val_summary_writer.add_scalar("best rrmse", best_mse, epoch)
            if opts.save_result:
                np.save(f"{data_save_path}/best_input_data.npy", np.array(input_data))
                np.save(f"{data_save_path}/best_output_data.npy", np.array(output_data))
                np.save(f"{data_save_path}/best_clean_data.npy", np.array(clean_data))
                torch.save(model, f"{data_save_path}/best_{opts.denoise_network}.pth")

        print('correct_d: {:3d}, sum_d:{:.4f}, acc_d:{}'.format(correct_d.cpu().numpy(), sum_d, correct_d.cpu().numpy()/sum_d*1.0))
        print('epoch: {:3d}, train_loss:{:.4f}, test_rrmse: {:.4f}, acc: {:.4f}, snr: {:.4f}'.format(epoch, train_loss, test_rrmse, sum_acc, sum_snr))
        f.write('epoch: {:3d}, test_rrmse: {:.4f}, acc: {:.4f}, snr: {:.4f}'.format(epoch, test_rrmse, sum_acc, sum_snr) + "\n")

    #with open(os.path.join('./logs/Denoisenet/{}/{}_{}_{}.log'.format(opts.noise_type, opts.denoise_network, w_c, w_f)), 'a+') as fp:
    #    fp.write('fold:{}, test_rrmse: {:.4f}, acc: {:.4f}, snr: {:.4f}'.format(fold, best_rrmse, best_acc, best_snr) + "\n")
    
    if opts.save_result:
        np.save(f"{data_save_path}/last_input_data.npy", test_data.EEG_data)
        np.save(f"{data_save_path}/last_output_data.npy", np.array(output_data))
        np.save(f"{data_save_path}/last_clean_data.npy", np.array(clean_data))
        torch.save(model, f"{data_save_path}/last_{opts.denoise_network}.pth")

if __name__ == '__main__':
    opts = get_opts()
    np.random.seed(0)
    opts.epochs = 200        # 50 200
    opts.depth = 6
    opts.noise_type = 'Hybrid'     # EMG EOG Hybrid
    opts.denoise_network = 'BG'

    opts.EEG_path = "./data/EEG_shuffle.npy"
    opts.NOS_path = f"./data/{opts.noise_type}_shuffle.npy"
    opts.save_path = "./results_SNR/{}/{}/".format(opts.noise_type, opts.denoise_network)

    for fold in range(10):
        print(f"fold:{fold}")
        model = pick_models(opts, data_num=512, embedding=1)
        print(opts.denoise_network)
        #summary(model, (3400,512))
        
        foldername = '{}_{}_{}_{}_{}_{}'.format(opts.denoise_network, opts.noise_type, opts.epochs, fold, w_c, w_f)
            
        train_log_dir = opts.save_path +'/'+foldername +'/'+ '/train'
        val_log_dir = opts.save_path +'/'+foldername +'/'+ '/test'
        data_save_path = opts.save_path +'/'+foldername +'/'
        
        if not os.path.exists(train_log_dir):
            os.makedirs(train_log_dir)
        
        if not os.path.exists(val_log_dir):
            os.makedirs(val_log_dir)
        
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)

        train(opts, model, train_log_dir, val_log_dir, data_save_path, fold)