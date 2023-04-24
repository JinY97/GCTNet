'''
Author: JinYin
Date: 2023-04-24 16:48:20
LastEditors: JinYin
LastEditTime: 2023-04-24 18:28:22
FilePath: \GCTNet\CalWSNR.py
'''
import os
import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data


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

    return out_file, in_file

# calculate E_k and H_k   
def cal_e(wave):
    E_t = 0
    for i in range(len(wave)):
        E_t = np.sum(np.square(wave[i])) + E_t

    E_k, H_k = [], []
    for i in range(len(wave)):
        E_k.append(np.sum(np.square(wave[i]))/E_t)
        
    for e in E_k:
        H_k.append(np.log(e) * e * (-1))
    
    return E_k, H_k
    
def cal_SNR(predict, truth):
    PS = np.sum(np.square(truth))  # power of signal
    PN = np.sum(np.square((predict - truth)))  # power of noise
    ratio = PS / PN
    return 10 * np.log10(ratio)

def cal_wsnr_wcc(denoised_EEG, clean_EEG):
    WSNR_e, WSNR_h, WCC_e, WCC_h = [], [], [], []
    for i in range(denoised_EEG.shape[0]):
        data1, data2 = denoised_EEG[i], clean_EEG[i]
        wave1 =pywt.wavedec(data1, "db9", level=5)
        wave2 =pywt.wavedec(data2, "db9", level=5)       
    
        Ek, Hk = cal_e(wave2)
        
        y1a5 = pywt.waverec(np.multiply(wave1, [1, 0, 0, 0, 0, 0]).tolist(), "db9")       # approximate coefficients
        y1d5 = pywt.waverec(np.multiply(wave1, [0, 1, 0, 0, 0, 0]).tolist(), "db9")       # detail coefficients（level5）
        y1d4 = pywt.waverec(np.multiply(wave1, [0, 0, 1, 0, 0, 0]).tolist(), "db9")       # detail coefficients（level4）
        y1d3 = pywt.waverec(np.multiply(wave1, [0, 0, 0, 1, 0, 0]).tolist(), "db9")       # detail coefficients（level3）
        y1d2 = pywt.waverec(np.multiply(wave1, [0, 0, 0, 0, 1, 0]).tolist(), "db9")       # detail coefficients（level2）
        y1d1 = pywt.waverec(np.multiply(wave1, [0, 0, 0, 0, 0, 1]).tolist(), "db9")       # detail coefficients（level1）
        y1 = [y1a5, y1d5, y1d4, y1d3, y1d2, y1d1]       # denoised EEG
        
        y2a5 = pywt.waverec(np.multiply(wave2, [1, 0, 0, 0, 0, 0]).tolist(), "db9")       # approximate coefficients
        y2d5 = pywt.waverec(np.multiply(wave2, [0, 1, 0, 0, 0, 0]).tolist(), "db9")       # detail coefficients（level5）
        y2d4 = pywt.waverec(np.multiply(wave2, [0, 0, 1, 0, 0, 0]).tolist(), "db9")       # detail coefficients（level4）
        y2d3 = pywt.waverec(np.multiply(wave2, [0, 0, 0, 1, 0, 0]).tolist(), "db9")       # detail coefficients（level3）
        y2d2 = pywt.waverec(np.multiply(wave2, [0, 0, 0, 0, 1, 0]).tolist(), "db9")       # detail coefficients（level2）
        y2d1 = pywt.waverec(np.multiply(wave2, [0, 0, 0, 0, 0, 1]).tolist(), "db9")       # detail coefficients（level1）
        y2 = [y2a5, y2d5, y2d4, y2d3, y2d2, y2d1]       # clean EEG
        
        WSNR_e_individual, WSNR_h_individual, WCC_e_individual, WCC_h_individual = [], [], [], []
        for j in range(len(Ek)):
            WSNR_e_individual.append(cal_SNR(y1[j], y2[j]) * Ek[j])
            WSNR_h_individual.append(cal_SNR(y1[j], y2[j]) * Hk[j])
            WCC_e_individual.append(np.corrcoef(y1[j], y2[j])[0,1] * Ek[j])
            WCC_h_individual.append(np.corrcoef(y1[j], y2[j])[0,1] * Hk[j])

        WSNR_e.append(np.sum(np.array(WSNR_e_individual)))
        WSNR_h.append(np.sum(np.array(WSNR_h_individual)))
        WCC_e.append(np.sum(np.array(WCC_e_individual)))
        WCC_h.append(np.sum(np.array(WCC_h_individual)))
    return np.mean(np.array(WSNR_e)), np.mean(np.array(WSNR_h)), np.mean(np.array(WCC_e)), np.mean(np.array(WCC_h))
