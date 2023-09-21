import cv2
from face_detect import face_detect
import os
from tqdm import tqdm
import numpy as np
import h5py
import pandas as pd
from preprocessing.convert_utils import getLocFromVideo, getFaceList, WrapperCap

def convertPURE(dir_path, folder_list, dst_path, img_size, test_device='cuda:0'):
    fd = face_detect.FaceDetect(test_device=test_device)
    with tqdm(total=len(folder_list), position=0, ncols=80, desc=dir_path) as pbar:
        for folder in folder_list:
            curr_folder_list = os.listdir(dir_path + '/' + folder)
            for curr_folder in curr_folder_list:
                video_path = dir_path + '/' + folder + '/' + curr_folder + '/' + 'cv_camera_sensor_stream_handler.avi'
                label_path = dir_path + '/' + folder + '/' + curr_folder + '/' + 'cms50_stream_handler.xml'
                cap = cv2.VideoCapture(video_path)
                frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总帧数
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
                cap = cv2.VideoCapture(video_path)
                raw_video = getFaceList(cap, face_boxes, img_size, frame_total)
                cap.release()

                label = pd.read_xml(label_path)
                ppg_data = []
                hr_data = []
                for i in range(len(label)):
                    ppg_data.append(label.loc[i][2])
                    hr_data.append(label.loc[i][1])

                if len(ppg_data) != frame_total:
                    ppg_data = np.interp(
                        np.linspace(1, len(ppg_data), frame_total),
                        np.linspace(1, len(ppg_data), len(ppg_data)), ppg_data)
                    hr_data = np.interp(
                        np.linspace(1, len(hr_data), frame_total),
                        np.linspace(1, len(hr_data), len(hr_data)), hr_data)
                raw_video[np.isnan(raw_video)] = 0
                ppg_data[np.isnan(ppg_data)] = 0

                data = h5py.File(f"{dst_path}/{folder}_{curr_folder}.hdf5", "w")
                data.create_dataset('raw_video', data=raw_video)
                data.create_dataset('ppg_data', data=ppg_data)
                data.create_dataset('hr_data', data=hr_data)
                data.create_dataset('fs', data=fs)
                data.close()

            pbar.update(1)

