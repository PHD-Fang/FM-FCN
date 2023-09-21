import cv2
import pandas as pd

from face_detect import face_detect
import os
from tqdm import tqdm
import numpy as np
import h5py
import pandas as pd
from preprocessing.convert_utils import getLocFromVideo, getFaceList, WrapperCap

def convertRLAP(dir_path, folder_list, dst_path, img_size, test_device='cuda:0'):
    fd = face_detect.FaceDetect(test_device=test_device)
    with tqdm(total=len(folder_list), position=0, ncols=80, desc=dir_path) as pbar:
        for folder in folder_list:
            curr_folder_list = os.listdir(dir_path + '/' + folder)
            for curr_folder in curr_folder_list:
                curr_path = dir_path + '/' + folder + '/' + curr_folder
                video_path = curr_path + '/' + 'video_ZIP_MJPG.avi'
                cap = cv2.VideoCapture(video_path)
                frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总帧数
                fs = cap.get(cv2.CAP_PROP_FPS)  # 视频采样率

                try:
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

                    label_path = curr_path + '/' + 'BVP.csv'
                    label = pd.read_csv(label_path)
                    ppg_data = []
                    ppg_ts = []
                    for i in range(len(label)):
                        ppg_data.append(label.loc[i][1])
                        ppg_ts.append(label.loc[i][0])
                    frame_ts_path = curr_path + '/' + 'frames_timestamp.csv'
                    frame_ts_data = pd.read_csv(frame_ts_path)
                    frame_ts = []
                    for i in range(len(frame_ts_data)):
                        frame_ts.append(frame_ts_data.loc[i][1])

                    if len(ppg_data) != frame_total:
                        ppg_data = np.interp(frame_ts, ppg_ts, ppg_data)
                    raw_video[np.isnan(raw_video)] = 0
                    ppg_data[np.isnan(ppg_data)] = 0
                except:
                    import traceback
                    traceback.print_exc()
                    print(video_path)
                    continue

                data = h5py.File(f"{dst_path}/{folder}_{curr_folder}.hdf5", "w")
                data.create_dataset('raw_video', data=raw_video)
                data.create_dataset('ppg_data', data=ppg_data)
                data.create_dataset('fs', data=fs)
                data.close()

            pbar.update(1)