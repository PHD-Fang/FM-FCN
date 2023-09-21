import cv2
from face_detect import face_detect
import os
from tqdm import tqdm
import numpy as np
import h5py
from preprocessing.convert_utils import getLocFromVideo, getFaceList, WrapperCap
import scipy.io as sio

def convertMMPD(dir_path, folder_list, dst_path, img_size, test_device='cuda:0'):
    def get_information(information):
        light = ''
        if information[0] == 'LED-low':
            light = 1
        elif information[0] == 'LED-high':
            light = 2
        elif information[0] == 'Incandescent':
            light = 3
        elif information[0] == 'Nature':
            light = 4

        motion = ''
        if information[1] == 'Stationary' or information[1] == 'Stationary (after exercise)':
            motion = 1
        elif information[1] == 'Rotation':
            motion = 2
        elif information[1] == 'Talking':
            motion = 3
        elif information[1] == 'Walking':
            motion = 4

        exercise = ''
        if information[2] == 'True' or information[2] == True:
            exercise = 1
        elif information[2] == 'False' or information[2] == False:
            exercise = 2

        skin_color = information[3][0]

        gender = ''
        if information[4] == 'male':
            gender = 1
        elif information[4] == 'female':
            gender = 2

        glasser = ''
        if information[5] == 'True' or information[5] == True:
            glasser = 1
        elif information[5] == 'False' or information[5] == False:
            glasser = 2

        hair_cover = ''
        if information[6] == 'True' or information[6] == True:
            hair_cover = 1
        elif information[6] == 'False' or information[6] == False:
            hair_cover = 2

        makeup = ''
        if information[7] == 'True' or information[7] == True:
            makeup = 1
        elif information[7] == 'False' or information[7] == False:
            makeup = 2
        return light, motion ,exercise, skin_color, gender, glasser, hair_cover, makeup
    fd = face_detect.FaceDetect(test_device=test_device)
    with tqdm(total=len(folder_list), position=0, leave=True, ncols=80, desc=dir_path) as pbar:
        for folder in folder_list:
            mid_path = os.path.join(dir_path, folder)
            file_list = os.listdir(mid_path)
            for filename in file_list:
                f = sio.loadmat(mid_path + '/' + filename)
                light = f['light'].tolist()[0]                  # 灯光，'LED-low','LED-high','Incandescent','Nature'
                motion = f['motion'].tolist()[0]             # 动作，'Stationary','Rotation','Talking','Walking'
                exercise = f['exercise'].tolist()[0]          #
                skin_color = f['skin_color'].tolist()[0]
                gender = f['gender'].tolist()[0]
                glasser = f['glasser'].tolist()[0]
                hair_cover = f['hair_cover'].tolist()[0]
                makeup = f['makeup'].tolist()[0]
                information = [light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup]
                light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup = get_information(information)
                # if light == 4 or motion != 1 or exercise != 2 or skin_color != 3:
                #     continue
                info = f"light_{light}-motion_{motion}-exercise_{exercise}-skin_color_{skin_color}-gender_{gender}-glasser_{glasser}-hair_cover_{hair_cover}-makeup_{makeup}"
                video = f['video'] * 255
                video = video.astype(np.uint8)
                ppg_data = f['GT_ppg'].ravel()

                frame_total = video.shape[0]
                for i in range(len(video)):
                    video[i] = cv2.cvtColor(video[i], cv2.COLOR_RGB2BGR)
                cap = WrapperCap(video)
                face_boxes = getLocFromVideo(cap, fd)
                if len(face_boxes) < frame_total:
                    frame_total = len(face_boxes)

                cap = WrapperCap(video)
                raw_video = getFaceList(cap, face_boxes, img_size, frame_total)

                raw_video[np.isnan(raw_video)] = 0

                if len(ppg_data) != frame_total:
                    ppg_data = np.interp(
                        np.linspace(1, len(ppg_data), frame_total),
                        np.linspace(1, len(ppg_data), len(ppg_data)), ppg_data)
                data = h5py.File(f"{dst_path}/{folder}_{filename}-" + info + ".hdf5", "w")
                data.create_dataset('frames', data=len(raw_video))
                data.create_dataset('raw_video', data=raw_video)
                data.create_dataset('ppg_data', data=ppg_data)
                data.close()