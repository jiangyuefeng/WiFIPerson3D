import os
import numpy as np
import utils.visualize as visualize
from config_test import opt
from os.path import join
import cv2
import math
import matplotlib.pyplot as plt


class Vectormap():
    def __init__(self, data, gt_vectormap, video_name, frame_number):
        self.prediction = data.cpu().numpy().squeeze()
        self.gt_vectormap = gt_vectormap.numpy().squeeze()
        self.get_ori_img(video_name, frame_number)
        # self.prediction = self.gt_vectormap
        # self.get_vectormap(self.gt_vectormap)
        self.prediction = self.resize_pafs(self.prediction, (1280, 720))
        X, Y, U, V = self.get_vectormap(self.prediction)
        visualize.display_pafs(X, Y, U, V, self.frame)
        self.gt_vectormap = self.resize_pafs(self.gt_vectormap, (1280, 720))
        X, Y, U, V = self.get_vectormap(self.gt_vectormap)
        visualize.display_pafs(X, Y, U, V, self.frame)

    def get_ori_img(self, video_name, frame_number):
        video_name = 'AlphaPose_2020-1-4_four_video.avi'
        frame_number = int(frame_number[0])
        video_dir = join(opt.test_data, "res")
        video = cv2.VideoCapture(join(video_dir, video_name))
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        ret, frame = video.read()
        self.frame = frame

    def resize_pafs(self, pafs, target_size):
        #print(pafs.shape)
        mapholder = []
        for i in range(pafs.shape[0]):
            a = cv2.resize(np.array(pafs[i, :, :]), target_size)
            mapholder.append(a)
        mapholder = np.array(mapholder)
        #print(mapholder.shape)
        return mapholder

    def get_vectormap(self, pafs, number=None):
        from numpy import ma
        if number is None:
            for i in range(13):
                u = pafs[2*i, :, :] * -1
                # u = cv2.resize(u, (1280, 720))
                v = pafs[2*i + 1, :, :]
                # v = cv2.resize(v, (1280, 720))
                # print(i)
                # print(u.max(), v.max())
                x, y = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))
                M = np.zeros(u.shape, dtype='bool')
                M[u**2 + v**2 < 0.2 * 0.2] = True
                u = ma.masked_array(u, mask=M)
                v = ma.masked_array(v, mask=M)
                if i == 0:
                    U = u
                    V = v
                    X = x
                    Y = y
                else:
                    U = ma.dstack((U, u))
                    V = ma.dstack((V, v))
                    X = np.dstack((X, x))
                    Y = np.dstack((Y, y))
                # print(U.shape)
            U = U.transpose(2, 0, 1)
            V = V.transpose(2, 0, 1)
            X = X.transpose(2, 0, 1)
            Y = Y.transpose(2, 0, 1)

        else:
            U = pafs[number, :, :] * -1
            # U = cv2.resize(U, (1280, 720))
            V = pafs[number + 1, :, :]
            # V = cv2.resize(V, (1280, 720))
            # print(U.max(), V.max())
            X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))
            M = np.zeros(U.shape, dtype='bool')
            M[U**2 + V**2 < 0.2 * 0.2] = True
            U = ma.masked_array(U, mask=M)
            V = ma.masked_array(V, mask=M)

        # self.frame = cv2.resize(self.frame, (82, 64))
        return X, Y, U, V
        # visualize.display_pafs(X, Y, U, V, self.frame)

    def show_sp(self):
        predition = self.prediction
        gt = self.gt_vectormap

        self.visualize_result(predition)
        self.visualize_result(gt)

    def visualize_result(self, data):
        from numpy import ma
        fig = plt.figure()
        #print(data.shape)
        for index in range(13):
            U = data[2*index, :, :] * -1
            V = data[2*index + 1, :, :]
            X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))
            M = np.zeros(U.shape, dtype='bool')
            M[U ** 2 + V ** 2 < 0.2 * 0.2] = True
            U = ma.masked_array(U, mask=M)
            V = ma.masked_array(V, mask=M)
            ax = fig.add_subplot(6, 5, index + 1)
            ax.imshow(self.frame, alpha=0.5)
            s = 10
            Q = ax.quiver(X[::s, ::s],
                          Y[::s, ::s],
                          U[::s, ::s],
                          V[::s, ::s],
                          scale=50,
                          headaxislength=4,
                          alpha=.5,
                          width=0.001,
                          color='r')
