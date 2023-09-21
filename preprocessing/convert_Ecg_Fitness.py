import cv2
from face_detect import face_detect
import os
from tqdm import tqdm
import numpy as np
import h5py
from preprocessing.convert_utils import getLocFromVideo, getFaceList

def convertEcgFitness(dir_path, folder_list, dst_path, img_size, test_device='cuda:0'):
    sub_folder_list = ['01', '02', '03', '04', '05', '06']
    fd = face_detect.FaceDetect(test_device=test_device)
    with tqdm(total=len(folder_list), position=0, leave=True, ncols=80, desc=dir_path) as pbar:
        for folder in folder_list:
            mid_path = os.path.join(dir_path, folder)
            for sub_folder in sub_folder_list:
                curr_path = os.path.join(mid_path, sub_folder)
                if not os.path.exists(curr_path):
                    continue
                name_list = ['c920-1', 'c920-2']
                for name in name_list:
                    video_name = curr_path + '/' + name + '.avi'
                    cap = cv2.VideoCapture(video_name)
                    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    # 视频总帧数
                    fs = cap.get(cv2.CAP_PROP_FPS)  # 视频采样率

                    # ---------------------------------------------------------------------#
                    #   人脸检测，获取最中间的
                    # ---------------------------------------------------------------------#
                    face_boxes = getLocFromVideo(cap, fd)
                    cap.release()
                    if len(face_boxes) < frame_total:
                        frame_total = len(face_boxes)

                    # ---------------------------------------------------------------------#
                    #   截取人脸区域
                    # ---------------------------------------------------------------------#
                    cap = cv2.VideoCapture(video_name)
                    raw_video = getFaceList(cap, face_boxes, img_size, frame_total)
                    cap.release()

                    # ---------------------------------------------------------------------#
                    #   处理心率标签
                    # ---------------------------------------------------------------------#
                    label_name = curr_path + '/viatom-raw.csv'
                    with open(label_name, 'r') as f:
                        data = f.readlines()
                    raw_hr = []
                    for i in range(1, len(data)):
                        line = data[i].strip().split(',')
                        raw_hr.append([int(line[0]), float(line[1]), float(line[2])])
                    raw_hr = np.array(raw_hr)
                    record_name = curr_path + '/c920.csv'
                    with open(record_name, 'r') as f:
                        data = f.readlines()
                    valid_hr = []
                    valid_ecg = []
                    hr_ts = raw_hr[:, 0]
                    for i in range(len(data)):
                        line = data[i].strip().split(',')
                        ts = int(line[0])
                        idx = np.argsort(np.abs(ts - hr_ts))[0]
                        valid_hr.append(raw_hr[idx, 2])
                        valid_ecg.append(raw_hr[idx, 1])
                    valid_hr = np.array(valid_hr)
                    valid_ecg = np.array(valid_ecg)

                    valid_ecg -= np.mean(valid_ecg)
                    valid_ecg /= np.std(valid_ecg)
                    valid_ecg[np.isnan(valid_ecg)] = 0

                    raw_video[np.isnan(raw_video)] = 0
                    valid_ecg[np.isnan(valid_ecg)] = 0
                    valid_hr[np.isnan(valid_hr)] = 0

                    data = h5py.File(f"{dst_path}/{folder}_{sub_folder}_{name}.hdf5", "w")
                    data.create_dataset('raw_video', data=raw_video)
                    data.create_dataset('ecg_data', data=valid_ecg)
                    data.create_dataset('hr', data=valid_hr)
                    data.create_dataset('fs', data=fs)
                    data.close()
            pbar.update(1)