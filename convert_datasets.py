#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author:    wingniuqichao
@contact:   wingniuqichao@outlook.com
@version:   1.0.0
@file:      convert_datasets.py
@time:      2023/5/22 10:27
"""


import numpy as np
import os
import matplotlib.pyplot as plt
from preprocessing import convert_utils
import multiprocessing
import math
plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'STFangsong']
plt.rcParams['axes.unicode_minus'] = False  

def startConvert(func, dir_path, folder_list, dst_path, img_size, test_device):
    NUM_WORKERS = 8
    folder_num_per_work = math.ceil(len(folder_list) / NUM_WORKERS)
    manager = multiprocessing.Manager()
    proc_list = []
    for i in range(NUM_WORKERS):
        curr_folders = folder_list[i * folder_num_per_work:(i + 1) * folder_num_per_work]
        proc = multiprocessing.Process(target=func, args=(dir_path, curr_folders, dst_path, img_size, test_device))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()

    manager.shutdown()

def convert_data(dir_path, dst_path, data_type, img_size=128, test_device="cpu"):

    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    if data_type == 'UBFC-phys':
        from preprocessing.convert_UBFC_Phys import convertUBFCPhys
        func = convertUBFCPhys
        folder_list = ['s%d' % i for i in range(1, 57)]
    elif data_type == 'UBFC-rppg':
        from preprocessing.convert_UBFC_Rppg import convertUBFCRppg
        func = convertUBFCRppg
        folder_list = [folder for folder in os.listdir(dir_path) if folder.startswith('subject')]
    elif data_type == "EcgFitness":
        from preprocessing.convert_Ecg_Fitness import convertEcgFitness
        func = convertEcgFitness
        folder_list = ["%02d" % i for i in range(0, 17)]
    elif data_type == "MMPD":
        from preprocessing.convert_MMPD import convertMMPD
        func = convertMMPD
        folder_list = [folder for folder in os.listdir(dir_path) if folder.startswith('subject')]
    elif data_type == "PURE":
        from preprocessing.convert_PURE import convertPURE
        func = convertPURE
        folder_list = [folder for folder in os.listdir(dir_path) if ('-' in folder and '.' not in folder)]
    elif data_type == "RLAP":
        from preprocessing.convert_RLAP import convertRLAP
        func = convertRLAP
        folder_list = [folder for folder in os.listdir(dir_path) if ('p0' in folder and '.' not in folder)]
    elif data_type == "cohface":
        from preprocessing.convert_cohface import convertCohface
        func = convertCohface
        folder_list = [str(i) for i in range(1, 41)]
    else:
        raise ValueError(f'Not support {data_type} datasets！')

    startConvert(func, dir_path, folder_list, dst_path, img_size, test_device)


if __name__ == '__main__':
    # data_type = 'UBFC-rppg'
    # dir_path = r"H:\datasets\RPPG\UBFC-rppg"
    # dst_path = r"H:\datasets\RPPG\UBFC-rppg\proc_small"

    # data_type = 'UBFC-phys'
    # dir_path = r"H:\datasets\RPPG\UBFC-phys"
    # dst_path = r"H:\datasets\RPPG\UBFC-phys\proc"

    # data_type = 'EcgFitness'
    # dir_path = r"H:\datasets\RPPG\ecg_fitness"
    # dst_path = r"H:\datasets\RPPG\ecg_fitness\proc"

    # data_type = 'MMPD'
    # dir_path = r"H:\datasets\RPPG\MMPD"
    # dst_path = r"H:\datasets\RPPG\MMPD\proc"

    # data_type = 'PURE'
    # dir_path = r"H:\datasets\RPPG\PURE"
    # dst_path = r"H:\datasets\RPPG\PURE\proc_small"

    # data_type = 'RLAP'
    # dir_path = r"H:\datasets\RPPG\RLAP"
    # dst_path = r"H:\datasets\RPPG\RLAP\proc_small"

    data_type = 'cohface'
    dir_path = r"H:\datasets\RPPG\cohface"
    dst_path = r"H:\datasets\RPPG\cohface\proc_small"


    img_size = 72
    device = 'cpu'  # 'cpu' or 'cuda:0'
    convert_data(dir_path, dst_path, data_type, img_size, device)
