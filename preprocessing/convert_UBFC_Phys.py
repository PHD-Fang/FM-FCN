import cv2
from face_detect import face_detect
import os
from tqdm import tqdm
import numpy as np
import h5py
from preprocessing.convert_utils import getLocFromVideo, getFaceList

def convertUBFCPhys(dir_path, folder_list, dst_path, img_size, test_device='cuda:0'):
    '''
    处理UBFC-Phys数据集
    Args:
        dir_path:

    Returns:

    '''
    fd = face_detect.FaceDetect(test_device=test_device)
    with tqdm(total=len(folder_list), position=0, ncols=80, desc=dir_path) as pbar:
        for folder in folder_list:
            curr_path = os.path.join(dir_path, folder)
            for i in range(1, 4):
                video_name = f'vid_{folder}_T{i}.avi'
                label_name = f'bvp_{folder}_T{i}.csv'
                cap = cv2.VideoCapture(os.path.join(curr_path, video_name))
                frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    # 视频总帧数
                fs = cap.get(cv2.CAP_PROP_FPS)                          # 视频采样率

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
                cap = cv2.VideoCapture(os.path.join(curr_path, video_name))
                raw_video = getFaceList(cap, face_boxes, img_size, frame_total)
                cap.release()

                # ---------------------------------------------------------------------#
                #   读取PPG信息
                # ---------------------------------------------------------------------#
                label_data = np.loadtxt(os.path.join(curr_path, label_name), dtype=float).ravel()
                if len(label_data) != frame_total:
                    label_data = np.interp(
                        np.linspace(1, len(label_data), frame_total),
                        np.linspace(1, len(label_data), len(label_data)), label_data)
                raw_video[np.isnan(raw_video)] = 0
                label_data[np.isnan(label_data)] = 0

                data = h5py.File(f"{dst_path}/{folder}_{i}.hdf5", "w")
                data.create_dataset('raw_video', data=raw_video)
                data.create_dataset('ppg_data', data=label_data)
                data.create_dataset('fs', data=fs)
                data.close()

            pbar.update(1)