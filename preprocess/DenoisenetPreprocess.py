'''
Author: Yin Jin
Date: 2022-03-08 20:15:16
LastEditTime: 2023-04-24 17:40:52
LastEditors: JinYin
Description: 数据预处理
'''

import math
from matplotlib.axis import Axis
import torch
import numpy as np
import scipy.io
from opts import get_opts
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

def DivideDataset(epoch_num, fold=0):
    x = np.arange(epoch_num)
    kf = KFold(n_splits=10, shuffle=False)
    train_list, test_list = [], []
    for train_index, test_index in kf.split(x):
        train_list.append(train_index)
        test_list.append(test_index)
    train_id, test_id = train_list[fold], test_list[fold]
    len_val = int(len(train_id) * 0.1)
    val_id, train_id = train_id[:len_val], train_id[len_val:]
    return train_id, val_id, test_id
    
def load_data_ECG(EEG_path, NOS_path, fold):    
    EEG_data, NOS_data = np.load(EEG_path), np.load(NOS_path)
    EEG_data_all, NOS_data_all = EEG_data.copy(), NOS_data.copy()
    EEG_data = EEG_data[:min(EEG_data.shape[0], NOS_data.shape[0]), :]
    NOS_data = NOS_data[:min(EEG_data.shape[0], NOS_data.shape[0]), :]
    train_id, val_id, test_id = DivideDataset(EEG_data.shape[0], fold)
    
    EEG_train_data, NOS_train_data = EEG_data[train_id, :], NOS_data[train_id, :]
    EEG_val_data, NOS_val_data = EEG_data[val_id, :], NOS_data[val_id, :]
    EEG_test_data, NOS_test_data = EEG_data[test_id, :], NOS_data[test_id, :]
    
    NOS_train_data = np.concatenate((NOS_train_data, NOS_train_data), axis=0)
    EEG_train_data = np.concatenate((EEG_train_data, EEG_data_all[NOS_data.shape[0]::, :]), axis=0)

    return EEG_train_data, NOS_train_data, EEG_val_data, NOS_val_data, EEG_test_data, NOS_test_data

def load_data_hybrid(EEG_path, EMG_path, EOG_path, fold):    
    EEG_data, EMG_data, EOG_data = np.load(EEG_path), np.load(EMG_path), np.load(EOG_path)
    EEG_data, EMG_data, EOG_data = EEG_data[:EOG_data.shape[0], :], EMG_data[:EOG_data.shape[0], :], EOG_data[:EOG_data.shape[0], :]
    NOS_data = np.zeros_like(EOG_data)
    for i in range(EOG_data.shape[0]):
        NOS_data[i] = EMG_data[i] / np.std(EMG_data[i]) + EOG_data[i] / np.std(EOG_data[i])
        
    train_id, val_id, test_id = DivideDataset(EEG_data.shape[0], fold)
    EEG_train_data, NOS_train_data = EEG_data[train_id, :], NOS_data[train_id, :]
    EEG_val_data, NOS_val_data = EEG_data[val_id, :], NOS_data[val_id, :]
    EEG_test_data, NOS_test_data = EEG_data[test_id, :], NOS_data[test_id, :]
    
    return EEG_train_data, NOS_train_data, EEG_val_data, NOS_val_data, EEG_test_data, NOS_test_data

def load_data(EEG_path, NOS_path, fold): 
    EEG_data, NOS_data = np.load(EEG_path), np.load(NOS_path)
    EEG_data = EEG_data[:min(EEG_data.shape[0], NOS_data.shape[0]), :]
    NOS_data = NOS_data[:min(EEG_data.shape[0], NOS_data.shape[0]), :]
    train_id, val_id, test_id = DivideDataset(EEG_data.shape[0], fold)
    
    EEG_train_data, NOS_train_data = EEG_data[train_id, :], NOS_data[train_id, :]
    EEG_val_data, NOS_val_data = EEG_data[val_id, :], NOS_data[val_id, :]
    EEG_test_data, NOS_test_data = EEG_data[test_id, :], NOS_data[test_id, :]

    return EEG_train_data, NOS_train_data, EEG_val_data, NOS_val_data, EEG_test_data, NOS_test_data

class EEGwithNoise(object):
    def __init__(self, EEG_data, NOS_data, batch_size=128):
        super(EEGwithNoise, self).__init__()
        self.EEG_data, self.NOS_data, self.SNR_value = [], [], []
        for value in 10 ** (0.05 * np.linspace(-5.0, 5.0, num=11)):
            self.EEG_data.append(EEG_data), self.NOS_data.append(NOS_data)
            self.SNR_value.append(np.zeros(shape=(EEG_data.shape[0])) + value)
        self.EEG_data = np.concatenate(self.EEG_data, axis=0)
        self.NOS_data = np.concatenate(self.NOS_data, axis=0)
        self.SNR_value = np.concatenate(self.SNR_value, axis=0)
        self.batch_size = batch_size

    def len(self):
        return math.ceil(self.EEG_data.shape[0] / self.batch_size)

    def get_item(self, item):
        EEG_data = self.EEG_data[item, :]
        NOS_data = self.NOS_data[item, :]
        
        SNR_value = self.SNR_value[item]
        EEG_rms = np.sqrt(np.sum(EEG_data ** 2) / EEG_data.shape[0])
        NOS_rms = np.sqrt(np.sum(NOS_data ** 2) / NOS_data.shape[0])
        coe = EEG_rms / (NOS_rms * SNR_value)
        NOS_data = NOS_data * coe
        EEG_NOS_data = NOS_data + EEG_data
        EEG_data = EEG_data / np.std(EEG_NOS_data)      # 计算全局标准差
        EEG_NOS_data = EEG_NOS_data / np.std(EEG_NOS_data)
        
        return EEG_NOS_data, EEG_data

    def get_batch(self, batch_id):
        start_id, end_id = batch_id * self.batch_size, min((batch_id + 1) * self.batch_size, self.EEG_data.shape[0])
        EEG_NOS_batch, EEG_batch = [], []
        for item in range(start_id, end_id):
            EEG_NOS_data, EEG_data = self.get_item(item)
            EEG_NOS_batch.append(EEG_NOS_data), EEG_batch.append(EEG_data)
        EEG_NOS_batch, EEG_batch = np.array(EEG_NOS_batch), np.array(EEG_batch)
        return EEG_NOS_batch, EEG_batch

    def shuffle(self):
        self.EEG_data = self.EEG_data[np.random.permutation(self.EEG_data.shape[0]), :]
        self.NOS_data = self.NOS_data[np.random.permutation(self.NOS_data.shape[0]), :]
        self.SNR_value = 10 ** (np.random.uniform(-5, 5, (self.EEG_data.shape[0])) * 0.05)


