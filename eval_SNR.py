'''
Author: Yin Jin
Date: 2022-03-13 13:32:48
LastEditTime: 2023-10-26 16:41:58
LastEditors: JinYin
'''
import os
import numpy as np
import scipy.io as scio
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import torch

plt.style.use(['science'])
l_db = 11

def read_mat(mat_file_path):
    if os.path.exists(mat_file_path):
        return scio.loadmat(mat_file_path)

def read_npy(npy_path):
    npy_data = np.load(npy_path, allow_pickle=True)
    return npy_data

def read_data(out_file_path, in_file_path):
    output_file = read_npy(out_file_path)
    input_file = read_npy(in_file_path)

    out_file, in_file = output_file[0], input_file[0]
    for i in range(1, output_file.shape[0]):
        out_file = np.concatenate((out_file, output_file[i]))
        in_file = np.concatenate((in_file, input_file[i]))

    output_file, input_file = out_file, in_file
    return out_file, in_file

def cal_acc_and_mse(output_file, input_file):
    res_cc, res_rrmse = [], []
    res_snr = []
    for i in range(input_file.shape[0]):
        out_file = output_file[i, :]
        in_file = input_file[i, :]
            
        a = in_file     # clean
        b = out_file        # predicted
        
        res_cc.append((np.corrcoef(a, b)[0,1]))
        res_snr.append(cal_SNR(b, a))
        
        res_rrmse.append(mean_squared_error(a, b, squared=False) / mean_squared_error(a, np.zeros_like(a), squared=False))

        rrmse = np.mean(np.array(res_rrmse).reshape(-1))
    print("输出维度:{}，CC结果：{}".format(len(res_cc), np.mean(np.array(res_cc).reshape(-1))))
    print("输出维度:{}，RRMSE结果：{}".format(len(res_rrmse), rrmse))
    print("输出维度:{}，SNR结果：{}".format(len(res_snr), np.mean(np.array(res_snr).reshape(-1))))

def cal_SNR(predict, truth):
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()

    PS = np.sum(np.square(truth))  # power of signal
    PN = np.sum(np.square((predict - truth)))  # power of noise
    ratio = PS / PN
    return 10 * np.log10(ratio)

def cal_acc_each_db(output_file, input_file):
    RRMSE, CC, SNR = [], [], []
    for i in range(l_db):
        out_file = output_file[i * (output_file.shape[0]//l_db):(i+1) * (output_file.shape[0]//l_db) , :]
        in_file = input_file[i * (input_file.shape[0]//l_db):(i+1) * (input_file.shape[0]//l_db) , :]
        
        res_cc, res_rrmse, res_snr = [], [], []
        for j in range(out_file.shape[0]):
            a = in_file[j, :]       # clean
            b = out_file[j, :]      # predicted
            
            res_cc.append((np.corrcoef(a, b)[0,1]))
            res_rrmse.append(mean_squared_error(a, b, squared=False) / mean_squared_error(a, np.zeros_like(a), squared=False))
            res_snr.append(cal_SNR(b, a))

        rrmse = np.mean(np.array(res_rrmse).reshape(-1))
        cc = np.mean(np.array(res_cc).reshape(-1))
        snr = np.mean(np.array(res_snr).reshape(-1))
        
        CC.append(cc)
        RRMSE.append(rrmse)
        SNR.append(snr)
        
        print("输出维度:{}，CC结果：{}".format(len(res_cc), cc))
        print("输出维度:{}，RRMSE结果：{}".format(len(res_rrmse), rrmse))
    return CC, RRMSE, SNR

def plot_db_cc(y, name, color, label, title, mode, path, clear = True):
    if clear == True:
        plt.clf()
    x_data = [i for i in range(-5, 6, 1)]
    y = np.array(y)
    plt.plot(x_data,y,'.-',label = label)
    plt.legend(frameon=True)
    plt.ylabel(f'{mode}', fontsize=14)
    plt.xlabel('SNR/dB', fontsize=14)
    plt.savefig("{}/{}.png".format(path, name))

def plot_db_snr(y, name, color, label, title, mode, path, clear = True):
    if clear == True:
        plt.clf()
    x_data = [i for i in range(-5, 6, 1)]
    y = np.array(y)
    plt.plot(x_data,y,'.-',label = label)
    plt.legend(frameon=True)
    plt.ylabel('SNR/dB', fontsize=14)
    plt.xlabel('SNR/dB', fontsize=14)
    plt.savefig("{}/{}.png".format(path, name))

def plot_db_rrmse(y, name, color, label, title, mode, path, clear = True):
    if clear == True:
        plt.clf()
    x_data = [i for i in range(-5, 6, 1)]
    y = np.array(y)
    plt.plot(x_data,y,'.-',label = label)
    plt.ylabel(f'{mode}', fontsize=14)
    plt.xlabel('SNR/dB', fontsize=14)
    plt.savefig("{}/{}.png".format(path, name))

if __name__ == "__main__":
    artifact_type = "EOG" 
    mode = "cal_all_db"        # cal_each_db cal_all_db

    dir_path = "./01_Denoisenet/CompareWorks/EOG/"
    save_path = "./Results/"
    model_dir_path_1 = dir_path + "/FCNN/FCNN_{}_200_mse_0_0.0/".format(artifact_type)
    model_dir_path_2 = dir_path + "/SimpleCNN/SimpleCNN_{}_200_mse_0_0.0/".format(artifact_type)
    model_dir_path_3 = dir_path + "/ResCNN/ResCNN_{}_200_mse_0_1/".format(artifact_type)
        
    model_output_file_path = model_dir_path_1 + "best_output_data.npy" 
    model_input_file_path = model_dir_path_1 + "best_clean_data.npy"

    model_output_file_path1 = model_dir_path_2 + "best_output_data.npy" 
    model_input_file_path1 = model_dir_path_2 + "best_clean_data.npy"
    
    model_output_file_path2 = model_dir_path_3 + "best_output_data.npy" 
    model_input_file_path2 = model_dir_path_3 + "best_clean_data.npy"
    
    model_out_file, model_in_file = read_data(model_output_file_path, model_input_file_path)
    model_out_file1, model_in_file1 = read_data(model_output_file_path1, model_input_file_path1)
    model_out_file2, model_in_file2 = read_data(model_output_file_path2, model_input_file_path2)

    if mode == "cal_all_db":
        cal_acc_and_mse(model_out_file, model_in_file)
        
    elif mode == "cal_each_db":     
        CC, RRMSE, SNR = cal_acc_each_db(model_out_file, model_in_file)
        CC1, RRMSE1, SNR1 = cal_acc_each_db(model_out_file1, model_in_file1)
        CC2, RRMSE2, SNR2 = cal_acc_each_db(model_out_file2, model_in_file2)

        cc_title = "The result of EOG from denoise net dataset"
        rrmse_title = "The result of EOG from denoise net dataset"
        snr_title = "The result of EOG from denoise net dataset"

        plot_db_cc(CC, f"{artifact_type}_cc", "g", f"FCNN", cc_title, "CC", save_path, True)
        plot_db_cc(CC1, f"{artifact_type}_cc", "g", f"SimpleCNN", cc_title, "CC", save_path, False)
        plot_db_cc(CC2, f"{artifact_type}_cc", "b", f"ResCNN", cc_title,  "CC", save_path, False)
        
        plot_db_rrmse(RRMSE, f"{artifact_type}_rrmse", "g", f"FCNN", rrmse_title, "RRMSE", save_path, True)
        plot_db_rrmse(RRMSE1, f"{artifact_type}_rrmse", "g", f"SimpleCNN", rrmse_title, "RRMSE", save_path, False)
        plot_db_rrmse(RRMSE2, f"{artifact_type}_rrmse", "b", f"ResCNN", rrmse_title,  "RRMSE", save_path, False)
        
        plot_db_snr(SNR, f"{artifact_type}_snr", "g", f"FCNN", snr_title, "SNR", save_path, True)
        plot_db_snr(SNR1, f"{artifact_type}_snr", "g", f"SimpleCNN", snr_title, "SNR", save_path,  False)
        plot_db_snr(SNR2, f"{artifact_type}_snr", "b", f"ResCNN", snr_title,  "SNR", save_path, False)
        

        
            

