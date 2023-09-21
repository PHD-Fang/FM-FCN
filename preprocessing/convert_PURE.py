import cv2
from face_detect import face_detect
import os
from tqdm import tqdm
import numpy as np
import h5py
from preprocessing.convert_utils import getLocFromVideo, getFaceList, WrapperCap
import json

def convertPURE(dir_path, folder_list, dst_path, img_size, test_device='cuda:0'):
    fd = face_detect.FaceDetect(test_device=test_device)
    with tqdm(total=len(folder_list), position=0, ncols=80, desc=dir_path) as pbar:
        for folder in folder_list:
            image_dir_path = dir_path + '/' + folder + '/' + folder
            label_file = dir_path + '/' + folder + '/' + folder + '.json'
            image_list = os.listdir(image_dir_path)
            image_list.sort()
            frame_total = len(image_list)  # 视频总帧数
            width = 640  # 视频图像宽度
            height = 480
            video = np.empty((frame_total, height, width, 3), dtype=np.uint8)
            for i in range(len(image_list)):
                image = cv2.imread(image_dir_path + '/' + image_list[i], 1)
                video[i, :, :, :] = image

            cap = WrapperCap(video)
            face_boxes = getLocFromVideo(cap, fd)
            if len(face_boxes) < frame_total:
                frame_total = len(face_boxes)
            cap = WrapperCap(video)
            # 确定人脸区域尺寸
            raw_video = getFaceList(cap, face_boxes, img_size, frame_total)

            ppg_signal = []
            ppg_time = []
            hr = []
            image_time = []
            with open(label_file) as json_file:
                json_data = json.load(json_file)
                for data in json_data['/FullPackage']:
                    ppg_signal.append(data['Value']['waveform'])
                    ppg_time.append(data['Timestamp'])
                    hr.append(data['Value']['pulseRate'])
                for data in json_data['/Image']:
                    image_time.append(data['Timestamp'])

            ppg_data = np.interp(image_time, ppg_time, ppg_signal)
            hr_data = np.interp(image_time, ppg_time, hr)

            ppg_data[np.isnan(ppg_data)] = 0

            data = h5py.File(f"{dst_path}/{folder}.hdf5", "w")
            data.create_dataset('frames', data=len(raw_video))
            data.create_dataset('raw_video', data=raw_video)
            data.create_dataset('ppg_data', data=ppg_data)
            data.create_dataset('hr', data=hr_data)
            data.close()
            pbar.update(1)