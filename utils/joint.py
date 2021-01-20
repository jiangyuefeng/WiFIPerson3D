from scipy.ndimage.filters import gaussian_filter
import numpy as np
import cv2
import math
from os.path import join
from config_test import opt
import matplotlib.pyplot as plt
import matplotlib
import scipy.io as sio


class Joint():
    def get_ori_img(self, video_name, frame_number):
        video_name = video_name[0] + '.avi'
        frame_number = int(frame_number[0])
        video_dir = join(opt.test_data, "frame_data")
        video = cv2.VideoCapture(join(video_dir, video_name))
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        ret, frame = video.read()
        self.oriImg = frame
    
    def zero_pad(self, oriData):
        oriData = oriData[2:-2, 2:-2, :]
        oriData = np.pad(oriData, ((2, 1), (2, 2), (0, 0)), 'constant')
        pading = np.zeros(oriData.shape)
        threshold = np.max(oriData) * 0.16
        heatmap = np.where(oriData < threshold, pading, oriData)    
        # heatmap = np.where(np.logical_and(heatmap > 0.99, heatmap != 1), pading, heatmap)
        return heatmap

    def __init__(self, heatmap, pafs, video_name, frame_number, pad=False):
        self.get_ori_img(video_name, frame_number)
        self.all_peaks = []
        self.peak_counter = 0
        heatmap = heatmap.cpu().numpy().squeeze()
        heatmap = heatmap.transpose(1, 2, 0)
        
        if pad:
            print('pad')
            heatmap = self.zero_pad(heatmap)
        
        for part in range(26 - 1):
            map_ori = heatmap[:, :, part]
            map = gaussian_filter(map_ori, sigma=0.3)

            map_left = np.zeros(map.shape)
            map_left[1:, :] = map[:-1, :]
            map_right = np.zeros(map.shape)
            map_right[:-1, :] = map[1:, :]
            map_up = np.zeros(map.shape)
            map_up[:, 1:] = map[:, :-1]
            map_down = np.zeros(map.shape)
            map_down[:, :-1] = map[:, 1:]

            times = 720 / 46.0
            # times = 1
            peaks_binary = np.logical_and.reduce(
                (map >= map_left, map >= map_right, map >= map_up,
                 map >= map_down, map > 0.1))
            peaks = list(
                zip(np.nonzero(peaks_binary)[1],
                    np.nonzero(peaks_binary)[0]))
            peaks_with_score = [(int(x[0] * times), int(x[1] * times),) + (map_ori[x[1], x[0]],) for x in peaks]
            id = range(self.peak_counter, self.peak_counter + len(peaks))
            peak_with_score_and_id = [
                peaks_with_score[i] + (id[i], ) for i in range(len(id))
            ]

            self.all_peaks.append(peak_with_score_and_id)
            self.peak_counter += len(peaks)
        print(self.all_peaks[24][-1][-1])

        pafs = pafs.cpu().numpy().squeeze()
        limbSeq = [[2, 9], [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8],
                   [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
                   [2, 1], [1, 16], [16, 18], [1, 17], [17, 19], [3, 18],
                   [6, 19], [15, 20], [20, 21], [15, 22], [12, 23], [23, 24],
                   [12, 25]]
        # mapIdx = [[0, 1], [14, 15], [22, 23], [16, 17], [18, 19], [24, 25],
        #           [26, 27], [6, 7], [2, 3], [4, 5], [8, 9], [10, 11], [12, 13],
        #           [30, 31], [32, 33], [36, 37], [34, 35], [38, 39], [20, 21],
        #           [28, 29], [40, 41], [42, 43], [44, 45], [46, 47], [48, 49],
        #           [50, 51]]
        mapIdx = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13],
                  [14, 15], [16, 17], [18, 19], [20, 21], [22, 23], [24, 25],
                  [26, 27], [28, 29], [30, 31], [32, 33], [34, 35], [36, 37],
                  [38, 39], [40, 41], [42, 43], [44, 45], [46, 47], [48, 49],
                  [50, 51]]
        self.connection_all = []
        self.special_k = []
        self.mid_num = 15

        mapholder = []
        for i in range(pafs.shape[0]):
            a = cv2.resize(np.array(pafs[i, :, :]), (1280, 720))
            mapholder.append(a)
        mapholder = np.array(mapholder)
        pafs = mapholder.transpose(1, 2, 0)
        print(pafs.shape)
        for k in range(len(mapIdx)):
            score_mid = pafs[:, :, [x for x in mapIdx[k]]]
            candA = self.all_peaks[limbSeq[k][0] - 1]
            candB = self.all_peaks[limbSeq[k][1] - 1]
            nA = len(candA)
            nB = len(candB)
            indexA, indexB = limbSeq[k]
            if nA != 0 and nB != 0:
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        if norm == 0:
                            continue
                        vec = np.divide(vec, norm)

                        startend = list(
                            zip(np.linspace(candA[i][0], candB[j][0], num=self.mid_num),
                                np.linspace(candA[i][1], candB[j][1], num=self.mid_num)))
                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] for I in range(len(startend))])
                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * self.oriImg.shape[0] / norm - 1, 0)
                        criterion1 = len(np.nonzero(score_midpts > 0.05)[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append([
                                i, j, score_with_dist_prior,
                                score_with_dist_prior + candA[i][2] + candB[j][2]
                            ])
                connection_candidate = sorted(connection_candidate,
                                              key=lambda x: x[2],
                                              reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        connection = np.vstack(
                            [connection, [candA[i][3], candB[j][3], s, i, j]])
                        if len(connection) >= min(nA, nB):
                            break
                self.connection_all.append(connection)
            else:
                self.special_k.append(k)
                self.connection_all.append([])

        self.subset = -1 * np.ones((0, 27))
        self.candidate = np.array([item for sublist in self.all_peaks for item in sublist])
        # print(self.candidate)
        # print(self.connection_all)

        for k in range(len(mapIdx)):
            if k not in self.special_k:
                partAs = self.connection_all[k][:, 0]
                partBs = self.connection_all[k][:, 1]
                indexA, indexB = np.array(limbSeq[k]) - 1

                for i in range(len(self.connection_all[k])):
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(self.subset)):
                        if self.subset[j][indexA] == partAs[i] or self.subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1
                    if found == 1:
                        j = subset_idx[0]
                        if self.subset[j][indexB] != partBs[i]:
                            self.subset[j][indexB] = partBs[i]
                            self.subset[j][-1] += 1
                            self.subset[j][-2] += self.candidate[
                                partBs[i].
                                astype(int), 2] + self.connection_all[k][i][2]
                    elif found == 2:
                        j1, j2 = subset_idx
                        membership = ((self.subset[j1] >= 0).astype(int) + (self.subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:
                            self.subset[j1][:-2] += (self.subset[j2][:-2] + 1)
                            self.subset[j1][-2:] += self.subset[j2][-2:]
                            self.subset[j1][-2] += self.connection_all[k][i][2]
                            self.subset = np.delete(self.subset, j2, 0)
                        else:
                            self.subset[j1][indexB] = partBs[i]
                            self.subset[j1][-1] += 1
                            self.subset[j1][
                                -2] += self.candidate[partBs[i].astype(
                                    int), 2] + self.connection_all[k][i][2]

                    elif not found and k < 24:
                        row = -1 * np.ones(27)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(
                            self.candidate[self.connection_all[k][i, :2].astype(int), 2]
                        ) + self.connection_all[k][i][2]
                        self.subset = np.vstack([self.subset, row])

        deleteIdx = []
        for i in range(len(self.subset)):
            if self.subset[i][-1] < 4 or self.subset[i][-2] / self.subset[i][-1] < 0.4:
                deleteIdx.append(i)
        self.subset = np.delete(self.subset, deleteIdx, axis=0)
        print(self.subset)

        colors = [[255, 0, 0], [255, 51, 0], [255, 102, 0], [255, 153, 0],
                  [255, 204, 0], [255, 255, 0], [204, 255, 0], [153, 255, 0],
                  [102, 255, 0], [51, 255, 0], [0, 255, 0], [0, 255, 51],
                  [0, 255, 102], [0, 255, 153], [0, 255, 204], [0, 255, 255],
                  [0, 204, 255], [0, 153, 255], [0, 102, 255], [0, 51, 255],
                  [0, 0, 255], [51, 0, 255], [102, 0, 255], [153, 0, 255],
                  [204, 0, 255], [255, 0, 255]]
        cmap = matplotlib.cm.get_cmap('hsv')
        canvas = self.oriImg

        for i in range(25):
            rgba = np.array(cmap(1 - i/18. - 1./36))
            rgba[0:3] *= 255
            for j in range(len(self.all_peaks[i])):
                cv2.circle(canvas, self.all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
        plt.figure()
        to_plot = cv2.addWeighted(self.oriImg, 0.3, canvas, 0.7, 0)
        plt.imshow(to_plot[:, :, [2, 1, 0]])
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(12, 12)

        stickwidth = 4
        for i in range(26):
            for n in range(len(self.subset)):
                index = self.subset[n][np.array(limbSeq[i]) - 1]
                if -1 in index:
                    continue
                cur_canvas = canvas.copy()
                Y = self.candidate[index.astype(int), 0]
                X = self.candidate[index.astype(int), 1]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1])**2 + (Y[0] - Y[1])**2)**0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)),
                                           (int(length / 2), stickwidth),
                                           int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

        plt.imshow(canvas[:, :, [2, 1, 0]])
        fig = matplotlib.pyplot.gcf()
        # fig.set_size_inches(12, 12)
