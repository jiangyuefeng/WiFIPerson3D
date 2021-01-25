# coding=UTF-8
import os
import numpy as np
from torch.utils import data
import cv2 as cv
# from PIL import Image
from torchvision import transforms as T
from os.path import join
from .prepare import *
import json
# from os.path import exists, join, split, dirname
# import sys
from torch.utils.data import DataLoader
# sys.path.insert(0, "lib/")
# sys.path.insert(0, "../lib/")

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
        #print(self.csi_dat)
        while (len(
                np.array(self.csi_dat[self.data_list[index].split(" ")[0][:-6]]
                         [str(start_time):str(next_frame_time)])) < 5):
            next_frame_time = next_frame_time + offsets.DateOffset(
                microseconds=50000)

        csi_temp = self.csi_dat[self.data_list[index].split(
            " ")[0][:-6]][str(start_time):str(next_frame_time)][:5]
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
        #print(sample_csi.shape)
        #print(self.root_dir)
        #print(self.data_list[index].split(" ")[:1][0])
        #print(type(self.data_list[index].split(" ")[:1][0]))
        sample_SM = sio.loadmat(
            join(self.root_dir, "mask_resize",
                self.data_list[index].split(" ")[:1][0]+"_"+str(index+2)) + ".mat")['masks']
        sample_SM = self.pose_transform(sample_SM)
        #转变float64  torch.size(1,46,82)
        sample_SM = sample_SM.double()
        #print(sample_SM.shape)
        json_dir = join(self.root_dir, 'res', 'alphapose_results.json')
        with open(json_dir) as f:
            # print(f)
            sample = json.load(f)
            JHM = sample[index]["keypoints"]
            JHM = np.array(JHM)
            sample_JHMs = get_heatmap((1280,720), JHM,(82, 46))[:, :, :-1]
            sample_PAFs = get_vectormap((1280,720), JHM, (82, 46))
            '''
            x = sample_PAFs.transpose((2,0,1))
            print(x[0])
            cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE)
            mask = 255*x[10]
            mask = mask.astype(np.uint8)
            cv.imshow('input_image', mask)
            cv.waitKey(0)
            cv.destroyAllWindows()
            '''
        sample_PAFs = self.pose_transform(sample_PAFs)
        sample_JHMs = self.pose_transform(sample_JHMs)
        sample_PAFs = sample_PAFs.double()
        sample_JHMs = sample_JHMs.double()
        #print(sample_JHMs.shape)
        #print(sample_PAFs.shape)
        # 转变float64 sample_JHMs torch.size(17,46,82) sample_PAFs torch.size(36,46,82)
        video_name = self.data_list[index].split(" ")[0]
        frame_number = self.data_list[index].split(" ")[1]

        sample = {
            'csi': sample_csi,
            'JHMs': sample_JHMs,
            'SM': sample_SM,
            'video': video_name,
            'PAFs': sample_PAFs,
            'frame': frame_number
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.data_list)

    def get_list(self):
        for root, dirs, files in os.walk(join(self.root_dir, "video")):
            for name in files:
                if os.path.splitext(name)[-1] == '.txt':
                    if "four" in name:
                        frame_index = 0
                        for line in open(join(root, name), 'r'):
                            if os.path.exists(
                                    join(
                                        self.root_dir, "res","alphapose_"+"results" + ".json")):
                                self.data_list.append(
                                    name.strip("Timestamp.txt") + 'video' + " " + str(frame_index) + " " + line.strip())
                            frame_index += 1
                        print(self.data_list[0])
        csi = dict()
        for root, dirs, files in os.walk(join(self.root_dir, "csi")):
            for name in files:

                if os.path.splitext(name)[-1] == '.txt':
                    if "four" in name:
                        # print(name.strip("time.txt"))
                        time_list = [
                            line.strip() for line in open(join(root, name), 'r')
                        ]
                        pd.to_datetime(time_list)
                        csi_data = sio.loadmat(
                            join(root, name.strip("time.txt") + "log.mat"))['video']
                        # for csi_data in csi:
                        #     print(csi_data[0][0][0][-1])

                        #print(csi_data)
                        # print(csi_data.shape[csi_data.ndim-1])
                        # print(csi_data[0][0][0][0][-1])
                        # bfee = Bfee.from_file(join(root, "_".join(name.split("_")[:-1]) + "_log.dat"), model_name_encode="gb2312")
                        # print(bfee.dicts[0]['csi'])
                        # print("_".join(name.split("_")[0:2]))
                        csi["_".join(name.split("_")[0:2])] = pd.Series(
                            (csi_data[:, :, :, index]
                             for index in range(csi_data.shape[csi_data.ndim - 1])),
                            index=time_list)
        self.csi_dat = csi

        # print(self.csi_dat)



if __name__ == '__main__':

    data = CSIList(root_dir="/home/public/b509/code/g19/jyf/mypro/PersonWiFi3/dataset")
    data[0]
    #data = csi.get_data()
    #print(data[0])
    # for i in range(len(data)):
    #     print(i)
    #     data[i]
    '''
    dataloader = DataLoader(
        data,
        batch_size=4,
        shuffle=True,
        num_workers=4
    )

    for batch, sample_batched in enumerate(dataloader):
        print(batch)

        if batch == 3:
             break
    '''
