from scipy.ndimage.filters import gaussian_filter
import numpy as np
import cv2
import math
from os.path import join
from config import opt
import matplotlib.pyplot as plt
import matplotlib
import scipy.io as sio
import pathlib
import shutil
from utils.visualize import Visualizer,get_joint_dist,get_joint


class Joint():
    def get_ori_img(self, video_name, frame_number):
        video_dir = join(opt.test_data, "video", "20210417")
        self.video_name = video_name +'.avi'
        self.frame_number = int(frame_number)
        video = cv2.VideoCapture(join(video_dir, self.video_name))
        video.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number - 1)
        ret, frame = video.read()
        # self.frame = cv2.resize(frame, (82, 46), interpolation=cv2.INTER_AREA)
        self.frame = frame
    
    def zero_pad(self, oriData):
        oriData = oriData[2:-2, 2:-2, :]
        oriData = np.pad(oriData, ((2, 1), (2, 2), (0, 0)), 'constant')
        pading = np.zeros(oriData.shape)
        threshold = np.max(oriData) * 0.16
        heatmap = np.where(oriData < threshold, pading, oriData)    
        # heatmap = np.where(np.logical_and(heatmap > 0.99, heatmap != 1), pading, heatmap)
        return heatmap
    
    def vis_frame(self, frame, preds_all, maxvals_all):
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (17, 11), (17, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]

        color = [(0, 255, 255), (0, 191, 255),(0, 255, 102),(0, 77, 255), (0, 255, 0), #Nose, LEye, REye, LEar, REar
                    (77,255,255), (77, 255, 204), (77,204,255), (191, 255, 77), (77,191,255), (191, 255, 77), #LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                    (204,77,255), (77,255,204), (191,77,255), (77,255,191), (127,77,255), (77,255,127), (0, 255, 255)] #LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck

        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), 
                    (77,255,222), (77,196,255), (77,135,255), (191,255,77), (77,255,77), 
                    (77,222,255), (255,156,127), 
                    (0,127,255), (255,127,77), (0,77,255), (255,77,36)]
        img = frame
        height,width = img.shape[:2]
        img = cv2.resize(img,(int(width/2), int(height/2)))
        img = np.ones(img.shape, np.uint8)
        img[:,:,:] = 0
        #print(preds_all.shape[0])
        for num in range(preds_all.shape[0]):
            preds = preds_all[num] 
            #print(preds)
            maxvals = maxvals_all[num]
            new = (preds[5]+preds[6])/2
            if maxvals.ndim == 1:
                maxvals = np.expand_dims(maxvals,axis=1)
            preds = np.concatenate((preds,(preds[5]+preds[6]).reshape(1,2)/2),axis=0)
            maxvals = np.concatenate((maxvals,(maxvals[5]+maxvals[6]).reshape(1,1)/2),axis=0)
            part_line = {}
            # Draw keypoints
            for n in range(preds.shape[0]):
                if maxvals[n] <= 0.05:
                    continue
                cor_x, cor_y = int(preds[n, 0]), int(preds[n, 1])
                part_line[n] = (int(cor_x/2), int(cor_y/2))
                bg = img.copy()
                cv2.circle(bg, (int(cor_x/2), int(cor_y/2)), 4, color[n], -1)
                img = cv2.addWeighted(bg, 0.7, img, 0.3, 0)
            # Draw limbs
            for i, (start_p, end_p) in enumerate(l_pair):
                if start_p in part_line and end_p in part_line:
                    start_xy = part_line[start_p]
                    end_xy = part_line[end_p]
                    bg = img.copy()

                    X = (start_xy[0], end_xy[0])
                    Y = (start_xy[1], end_xy[1])
                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                    stickwidth = (maxvals[start_p] + maxvals[end_p]) + 1
                    polygon = cv2.ellipse2Poly((int(mX),int(mY)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
                    cv2.fillConvexPoly(bg, polygon, line_color[i])
                    #cv2.line(bg, start_xy, end_xy, line_color[i], (2 * (kp_scores[start_p] + kp_scores[end_p])) + 1)
                    img = cv2.addWeighted(bg, 0.7, img, 0.3, 0)
        img = cv2.resize(img,(width,height),interpolation=cv2.INTER_CUBIC)
        return img
    
    def sh_joint(self):
        cmap = matplotlib.cm.get_cmap('hsv')
        img = self.vis_frame(self.frame, self.preds, self.maxvals)
        plt.figure()
        plt.imshow(img[:, :, [2, 1, 0]])
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(12, 12)
    
    def save_img(self):
        img_dir = join(opt.test_data,"csi_out",'20210116',self.video_name)
        save_folder = pathlib.Path(img_dir)
        if save_folder.exists() == False:
            shutil.rmtree(str(save_folder), ignore_errors=True)
            save_folder.mkdir(parents=True, exist_ok=True)
        
        img = self.vis_frame(self.frame, self.preds, self.maxvals)
        save_path = f'{save_folder}/csi_{str(self.frame_number)}.png'
        cv2.imwrite(save_path, img)
        print(f'Joint saved in {save_path}')
        return save_path
 

    def __init__(self, heatmap, box, video_name, frame_number, pad=False):
        '''
            heatmap     (17,46,82)
        '''

        self.get_ori_img(video_name, frame_number)
        self.preds, self.maxvals = get_joint(heatmap, box)
        self.preds = self.preds *(720/46)
        self.video_name = video_name
        self.frame_number = frame_number