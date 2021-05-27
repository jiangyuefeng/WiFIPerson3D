import numpy as np
import utils.visualize as visualize
from config import opt
import cv2
from os.path import join
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import math


class Heatmap():
    def __init__(self, data, gt_heatmap, video_name, frame_number):
        self.origin_data = data.cpu().numpy().squeeze()
        self.gt_heatmap = gt_heatmap.numpy().squeeze()
        self.get_ori_img(video_name,frame_number)
        # self.get_joints()

    def zero_pad(self, oriData):
        oriData = oriData[:, 2:-2, 2:-2]
        oriData = np.pad(oriData, ((0, 0), (2, 2), (2, 2)), 'constant')
        pading = np.zeros(oriData.shape)
        threshold = np.max(oriData) * 0.16
        heatmap = np.where(oriData < threshold, pading, oriData)
        # heatmap = oriData
        # heatmap = np.where(np.logical_and(heatmap > 0.99, heatmap != 1), pading, heatmap)
        return heatmap

    def get_ori_img(self, video_name, frame_number):
        video_dir = join(opt.test_data, "video", "20210417")
        self.video_name = video_name +'.avi'
        self.frame_number = int(frame_number)
        video = cv2.VideoCapture(join(video_dir, self.video_name))
        video.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number - 1)
        ret, frame = video.read()
        # self.frame = cv2.resize(frame, (82, 46), interpolation=cv2.INTER_AREA)
        self.frame = frame

    def resize_data(self, oriData, target_size):
        mahpholder = []
        for i in range(oriData.shape[0]):
            a = cv2.resize(np.array(oriData[i, :, :]), target_size)
            mahpholder.append(a)
        mahpholder = np.array(mahpholder)
        targetData = mahpholder
        return targetData

    def get_joints(self):

        heatmap_avg = self.origin_data
        heatmap_avg = self.gt_heatmap.transpose(1, 2, 0)
        heatmap_avg = cv2.resize(heatmap_avg, (1280, 720))
        all_peaks = []
        peak_counter = 0

        for part in range(18 - 1):
            map_ori = heatmap_avg[:, :, part]

            map = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(map.shape)
            map_left[1:, :] = map[:-1, :]
            map_right = np.zeros(map.shape)
            map_right[:-1, :] = map[1:, :]
            map_up = np.zeros(map.shape)
            map_up[:, 1:] = map[:, :-1]
            map_down = np.zeros(map.shape)
            map_down[:, :-1] = map[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (map >= map_left, map >= map_right, map >= map_up,
                 map >= map_down, map > 0.1))
            peaks = list(
                zip(np.nonzero(peaks_binary)[1],
                    np.nonzero(peaks_binary)[0]))
            peaks_with_score = [x + (map_ori[x[1], x[0]], ) for x in peaks]
            id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [
                peaks_with_score[i] + (id[i], ) for i in range(len(id))
            ]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)
        print(peak_counter)

    def show_all(self):
        predition = self.origin_data
        predition = self.zero_pad(predition)
        visualize.display_heatmap(predition, self.gt_heatmap,
                                  self.frame)

    def show_sp(self):
        predition = self.origin_data
        predition = self.zero_pad(predition)
        predition = self.resize_data(predition, (1280, 720))
        gt = self.resize_data(self.gt_heatmap, (1280, 720))

        self.visualize_result(predition)
        self.visualize_result(gt)

    def visualize_result(self, data):
        fig = plt.figure()
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        for index in range(data.shape[0]):
            ax = fig.add_subplot(math.ceil(data.shape[0] / 5), 5, index + 1)
            divider = make_axes_locatable(ax)
            ax_cb = divider.new_horizontal(size="3%", pad=0.05)
            fig_index = ax.get_figure()
            fig_index.add_axes(ax_cb)
            ax.imshow(self.frame)
            im = ax.imshow(data[index, :, :], alpha=0.5)
            plt.colorbar(im, cax=ax_cb)
            ax_cb.yaxis.tick_right()
            ax_cb.yaxis.set_tick_params(labelright=True)
