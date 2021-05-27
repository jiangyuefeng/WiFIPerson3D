# coding=UTF-8
import os
import numpy as np
from torch.utils import data
import cv2 as cv
# from PIL import Image
from torchvision import transforms as T
from os.path import join
import time
import json
# from os.path import exists, join, split, dirname
# import sys
from torch.utils.data import DataLoader
# sys.path.insert(0, "lib/")
# sys.path.insert(0, "../lib/")
import matplotlib.pyplot as plt
import pandas as pd
from pandas import offsets
import scipy.io as sio
# from PIL import Image
# import cv2
# import math


class CSIList(data.Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.data_list = []
        self.sample_list = []
        self.list = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
        #self.list = [33]
        self.current_file = None
        self.bfee = None
        self.time_list = None
        self.current_csi_dat = None

        self.get_list()
        

        self.sm_transform = T.Compose([
            # T.Pad(padding=2, fill=0),
            T.Resize((46, 82)),
            T.ToTensor(),
        ])
        self.pose_transform = T.Compose([
            T.ToTensor(),
        ])

    def __getitem__(self, index):
        current_frame_time = pd.to_datetime(
            self.data_list[index].split(" ")[2] + " " + self.data_list[index].split(" ")[3])
        video_name = self.data_list[index].split(" ")[0]
        #print(video_name)
        frame_number = self.data_list[index].split(" ")[1]
        #print(frame_number)
        #print(current_frame_time)
        time_index = int(current_frame_time.microsecond / 50000)
        start_time = pd.to_datetime(
            str(current_frame_time)[:-7] + "." + str(time_index * 50000).zfill(6))
        #print(start_time)
        next_frame_time = start_time + offsets.DateOffset(microseconds=50000)
        #print(next_frame_time)
        # start_time = str(start_time.hour).zfill(2) + ":" + str(start_time.minute).zfill(2) + ":" + str(start_time.second).zfill(2) + "." + str(start_time.microsecond).zfill(6)
        # end_time = str(next_frame_time.hour).zfill(2) + ":" + str(next_frame_time.minute).zfill(2) + ":" + str(next_frame_time.second).zfill(2) + "." + str(next_frame_time.microsecond).zfill(6)

        # print("'start_time', {}, 'end_time', {}, 'len', {}".format(start_time, end_time, len(self.current_csi_dat[start_time:end_time])))
        #print(self.csi_dat[self.data_list[index].split(" ")[0][:-6]])
        #print(self.csi_dat[video_name[:-6]])
        i = 0
        while (len(np.array(self.csi_dat[video_name[:-6]]
                            [str(start_time):str(next_frame_time)])) < 5):
            #print(f"i:{i}")
            i = i + 1
            if i <= 10:
                next_frame_time = next_frame_time + offsets.DateOffset(microseconds=50000)
            else:
                start_time = start_time - offsets.DateOffset(microseconds=50000)    

        csi_temp = self.csi_dat[video_name[:-6]][str(start_time):str(next_frame_time)][:5]
        # print(csi_temp[0])
        # csi_temp = np.array(csi_temp)
        # print(np.concatenate(np.array(csi_temp), axis=2).shape)
        # csi_temp = np.absolute(np.concatenate(np.array(csi_temp)))
        # print(csi_temp.shape)
        # csi_temp = csi_temp.transpose(0 ,2, 1)
        # print(csi_temp.shape)
        # # sample_csi = np.absolute(csi_temp)
        # sample_csi = np.resize(csi_temp, (150, 3, 3))
        #print(np.concatenate(csi_temp, axis=2))
        sample_csi = np.concatenate(csi_temp, axis=2).transpose(2, 0, 1)
        #print(self.root_dir)
        #print(self.data_list[index].split(" ")[:1][0])
        #print(type(self.data_list[index].split(" ")[:1][0]))
        mat_data = sio.loadmat(join(self.root_dir, "alphapose", '20210417', 'mu_alphapose_resize', "_".join(
                self.data_list[index].split(" ")[:2])) + ".mat")
        sample_JHMs = mat_data['heatmap'][:,:,:-1]
        sample_box = mat_data['bbox'][:,:,:2]
        print(sample_box.shape)
        sample_box = self.pose_transform(sample_box)
        sample_JHMs = self.pose_transform(sample_JHMs)
        #print(sample_JHMs.shape)
        # 转变float64 sample_JHMs torch.size(17,46,82)

        sample = {
            'csi': sample_csi,
            'JHMs': sample_JHMs,
            'box': sample_box,
            'video': video_name,
            'frame': frame_number
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.data_list)

    def get_list(self):
        for root, dirs, files in os.walk(join(self.root_dir, "video", "20210417")):
            print(root)
            for frame_index in self.list:
            #if True:
                #frame_index = 1
                name = "2021-4-18_p"+str(frame_index)  +"_Timestamp.txt"
                frame_number = 0
                for line in open(join(root, name), 'r'):
                    if os.path.exists(
                            join(
                                self.root_dir, "alphapose",'20210417','mu_alphapose_resize',
                                "2021-4-18_p"+str(frame_index) + "_video_" + str(frame_number) + ".mat")) and frame_number > 0:
                        self.data_list.append(
                            "2021-4-18_p"+str(frame_index)  + '_video' + " " + str(frame_number) + " " + line.strip())
                    frame_number += 1
        # print(self.data_list)
        csi = dict()
        for root, dirs, files in os.walk(join(self.root_dir, "csi", "20210417")):
            for csi_index in self.list:
            #if True:
                #csi_index = 1
                csi_name = "t" + str(csi_index)+"_scaled"
                name = "t" + str(csi_index) +".txt"

                time_list = [
                    line.strip() for line in open(join(root, name), 'r')
                ]
                pd.to_datetime(time_list)
                csi_data = sio.loadmat(
                    join(self.root_dir,"mat", '20210417', csi_name + ".mat"))['video']
                #取摸
                csi_data = np.abs(csi_data)
                frame_name = "2021-4-18_p"+ str(csi_index)
                csi[frame_name] = pd.Series(
                    (csi_data[:, :, :, index]
                        for index in range(csi_data.shape[csi_data.ndim - 1])),
                    index=time_list)
        self.csi_dat = csi




if __name__ == '__main__':

    data = CSIList(root_dir="/home/public/b509/code/dataset")
    print(data[0])
    '''
    print(len(data))
    for i in range(len(data)):
        data[i]
    
    dataloader = DataLoader(
        data,
        batch_size=4,
        shuffle=False,
        num_workers=8,
    )

    for index, data in enumerate(dataloader):
        plt.figure()
        plt.imshow(data['JHMs'][0][1,:,:])
        plt.show()
    '''
