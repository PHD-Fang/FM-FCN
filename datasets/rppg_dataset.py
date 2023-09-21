import numpy as np
import torch
from torch.utils.data import Dataset
import os, cv2
import h5py


class RppgDataset(Dataset):
    def __init__(self, dir_path_list, img_size, length=1, overlap=0, st=0, ed=1, argment=False, mode='diff', diff_norm=False, video_norm_per_channel=True):
        self.m_dir_path_list = dir_path_list       
        self.m_img_size = img_size                 
        self.m_length = length
        self.m_overlap = overlap
        self.m_argment = argment
        self.m_mode = mode  # diff, cont, hr
        self.m_diff_norm = diff_norm
        self.m_video_norm_per_channel = video_norm_per_channel
        # ---------------------------------------------------------------------#
        #   get all files
        # ---------------------------------------------------------------------#
        self.m_file_list = []
        for dir_path in dir_path_list:
            curr_file_list = [dir_path + '/' + filename for filename in os.listdir(dir_path) if
                              filename.endswith('.hdf5')]
            curr_file_list.sort()
            curr_file_list = curr_file_list[int(st * len(curr_file_list)):int(ed * len(curr_file_list))]
            self.m_file_list.extend(curr_file_list)

        self.m_video_index = {}
        self.m_total_num = 0
        for filename in self.m_file_list:
            data = h5py.File(filename, "r")
            frames = data['raw_video'].shape[0] - 1
            num = int((frames - overlap) / (length - overlap))
            self.m_video_index[filename] = {'st': self.m_total_num, 'ed': self.m_total_num + num}
            self.m_total_num += num
            data.close()

    def getFileByIndex(self, index):
        aim_filename = ""
        for filename in self.m_video_index:
            if self.m_video_index[filename]['st'] <= index < self.m_video_index[filename]['ed']:
                aim_filename = filename
                break
        return aim_filename

    def __getitem__(self, index):
        index %= self.m_total_num
        filename = self.getFileByIndex(index)

        st = (index - self.m_video_index[filename]['st']) * (self.m_length - self.m_overlap)
        if self.m_argment and st >= (self.m_length - self.m_overlap):
            roll_num = int(np.random.random() * (self.m_length - self.m_overlap))
            st -= roll_num
        


        data = h5py.File(filename, "r")
        if self.m_mode in ['DIFF', 'BIG_SMALL', 'ONLY_DIFF']:
            
            raw_video = data['raw_video'][st:st + self.m_length + 1]
            # _, h, w, _ = raw_video.shape
            # raw_video = raw_video[:, h//2:, :, :]
            # gap = int(raw_video.shape[1]/6)
            # raw_video = raw_video[:, gap:-gap, gap:-gap, :]
            ppg_data = data['ppg_data'][st + 1:st + self.m_length + 1] - data['ppg_data'][st:st + self.m_length]
        elif self.m_mode == 'CONT':
            raw_video = data['raw_video'][st:st + self.m_length]
            ppg_data = data['ppg_data'][st:st + self.m_length]
        else:
            raw_video = data['raw_video'][st:st + self.m_length]
            hr_data = data['hr'][st + int(self.m_length / 2)]

        if self.m_argment:
            # ---------------------------------------------------------------------#
            #   rotation
            # ---------------------------------------------------------------------#
            h, w = raw_video.shape[1:3]
            center = (w // 2, h // 2)
            angle = np.random.random() * 60 - 30
            r = 0.5 * np.random.random() + 1
            M = cv2.getRotationMatrix2D(center, angle, r)
            for i in range(len(raw_video)):
                raw_video[i] = cv2.warpAffine(raw_video[i], M, (w, h))

            # ---------------------------------------------------------------------#
            #   flip
            # ---------------------------------------------------------------------#
            if np.random.random() > 0.5:
                raw_video = np.flip(raw_video, axis=2)
            if np.random.random() > 0.5:
                raw_video = np.flip(raw_video, axis=1)

            # ---------------------------------------------------------------------#
            #   Brightness and contrast adjustment
            # ---------------------------------------------------------------------#
            bri_mean = np.mean(raw_video)
            aa = np.random.random() + 0.5
            bb = np.random.random() * 60 - 30
            raw_video = aa * (raw_video - bri_mean) + bb + bri_mean

            # ---------------------------------------------------------------------#
            #   gaussian noise
            # ---------------------------------------------------------------------#
            # sigma = np.random.random() * 1
            # gauss = np.random.normal(0, sigma, raw_video.shape)
            # raw_video += gauss

        raw_video = np.clip(raw_video, 0, 255).astype(float)

        if self.m_mode == 'DIFF':
            if self.m_diff_norm == 'A-B':
                diff_video = raw_video[1:] - raw_video[:-1]
            else:
                diff_video = (raw_video[1:] - raw_video[:-1]) / (raw_video[1:] + raw_video[:-1] + 1e-7)
            raw_video = (raw_video[1:] + raw_video[:-1]) / 2
            video_data = np.empty((self.m_length, 2, self.m_img_size, self.m_img_size, 3), dtype=float)
            if raw_video.shape[0] != self.m_img_size:
                for i in range(self.m_length):
                    video_data[i, 1, :, :, :] = cv2.resize(diff_video[i], (self.m_img_size, self.m_img_size),
                                                           cv2.INTER_AREA)
                    video_data[i, 0, :, :, :] = cv2.resize(raw_video[i], (self.m_img_size, self.m_img_size),
                                                           cv2.INTER_AREA)
            else:
                video_data[:, 0, :, :, :] = raw_video[:, :, :, :]
                video_data[:, 1, :, :, :] = diff_video[:, :, :, :]

            # ---------------------------------------------------------------------#
            #   normalization
            # ---------------------------------------------------------------------#
            if self.m_video_norm_per_channel:
                for i in range(3):
                    video_data[:, 0, :, :, i] -= np.mean(video_data[:, 0, :, :, i])
                    video_data[:, 0, :, :, i] /= np.std(video_data[:, 0, :, :, i])
                    video_data[:, 1, :, :, i] -= np.mean(video_data[:, 1, :, :, i])
                    video_data[:, 1, :, :, i] /= np.std(video_data[:, 1, :, :, i])
            else:
                video_data[:, 0, :, :, :] -= np.mean(video_data[:, 0, :, :, :])
                video_data[:, 0, :, :, :] /= np.std(video_data[:, 0, :, :, :])
                video_data[:, 1, :, :, :] -= np.mean(video_data[:, 1, :, :, :])
                video_data[:, 1, :, :, :] /= np.std(video_data[:, 1, :, :, :])
            ppg_data -= np.mean(ppg_data)
            ppg_data /= np.std(ppg_data)
            video_data = torch.tensor(np.transpose(video_data, (0, 1, 4, 2, 3)), dtype=torch.float32).contiguous()
            ppg_data = torch.tensor(ppg_data, dtype=torch.float32)
        elif self.m_mode == 'ONLY_DIFF':
            if self.m_diff_norm == 'A-B':
                diff_video = raw_video[1:] - raw_video[:-1]
            else:
                diff_video = (raw_video[1:] - raw_video[:-1]) / (raw_video[1:] + raw_video[:-1] + 1e-7)

            video_data = np.empty((self.m_length, self.m_img_size, self.m_img_size, 3), dtype=float)
            if diff_video.shape[0] != self.m_img_size:
                for i in range(self.m_length):
                    video_data[i, :, :, :] = cv2.resize(diff_video[i], (self.m_img_size, self.m_img_size),
                                                           cv2.INTER_AREA)
            else:
                video_data[:, :, :, :] = diff_video[:, :, :, :]

            # ---------------------------------------------------------------------#
            #   normalization
            # ---------------------------------------------------------------------#
            if self.m_video_norm_per_channel:
                for i in range(3):
                    video_data[:, :, :, i] -= np.mean(video_data[:, :, :, i])
                    video_data[:, :, :, i] /= np.std(video_data[:, :, :, i])
            else:
                video_data[:, :, :, :] -= np.mean(video_data[:, :, :, :])
                video_data[:, :, :, :] /= np.std(video_data[:, :, :, :])
            ppg_data -= np.mean(ppg_data)
            ppg_data /= np.std(ppg_data)
            video_data = torch.tensor(np.transpose(video_data, (0, 3, 1, 2)), dtype=torch.float32).contiguous()
            ppg_data = torch.tensor(ppg_data, dtype=torch.float32)
        elif self.m_mode == 'BIG_SMALL':
            if self.m_diff_norm == 'A-B':
                diff_video = raw_video[1:] - raw_video[:-1]
            else:
                diff_video = (raw_video[1:] - raw_video[:-1]) / (raw_video[1:] + raw_video[:-1] + 1e-7)
            raw_video = (raw_video[1:] + raw_video[:-1]) / 2
            video_data = np.empty((self.m_length, self.m_img_size, self.m_img_size, 3), dtype=float)
            diff_data = np.empty((self.m_length, 9, 9, 3), dtype=float)

            for i in range(self.m_length):
                diff_data[i, :, :, :] = cv2.resize(diff_video[i], (9, 9),
                                                        cv2.INTER_AREA)
                video_data[i, :, :, :] = cv2.resize(raw_video[i], (self.m_img_size, self.m_img_size),
                                                        cv2.INTER_AREA)


            # ---------------------------------------------------------------------#
            #   normalization
            # ---------------------------------------------------------------------#
            if self.m_video_norm_per_channel:
                for i in range(3):
                    video_data[:, :, :, i] -= np.mean(video_data[:, :, :, i])
                    video_data[:, :, :, i] /= np.std(video_data[:, :, :, i])
                    diff_data[:, :, :, i] -= np.mean(diff_data[:, :, :, i])
                    diff_data[:, :, :, i] /= np.std(diff_data[:, :, :, i])

            else:
                video_data[:, :, :, :] -= np.mean(video_data[:, :, :, :])
                video_data[:, :, :, :] /= np.std(video_data[:, :, :, :])
                diff_data[:, :, :, :] -= np.mean(diff_data[:, :, :, :])
                diff_data[:, :, :, :] /= np.std(diff_data[:, :, :, :])
            ppg_data -= np.mean(ppg_data)
            ppg_data /= np.std(ppg_data)
            video_data = torch.tensor(np.transpose(video_data, (0, 3, 1, 2)), dtype=torch.float32).contiguous()
            diff_data = torch.tensor(np.transpose(diff_data, (0, 3, 1, 2)), dtype=torch.float32).contiguous()
            ppg_data = torch.tensor(ppg_data, dtype=torch.float32)
            video_data = [video_data, diff_data]
        elif self.m_mode == 'CONT':
            video_data = raw_video
            video_data -= np.mean(video_data)
            video_data /= np.std(video_data)
            ppg_data -= np.mean(ppg_data)
            ppg_data /= np.std(ppg_data)
            video_data = torch.tensor(np.transpose(video_data, (3, 0, 1, 2)), dtype=torch.float32).contiguous()
            ppg_data = torch.tensor(ppg_data, dtype=torch.float32)
        else:
            video_data = raw_video
            video_data -= np.mean(video_data)
            video_data /= np.std(video_data)
            hr_data /= 60.
            video_data = torch.tensor(np.transpose(video_data, (3, 0, 1, 2)), dtype=torch.float32).contiguous()
            ppg_data = torch.tensor(hr_data, dtype=torch.float32)

        data.close()
        return video_data, ppg_data

    def __len__(self):
        return self.m_total_num


