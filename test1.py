from config_test import opt
import models
import torch
import os
from data.dataset import CSIList
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable
from utils.mask import Mask
import numpy

class Test():
    def __init__(self):
        self.model = getattr(models, 'WiFiModel3')().eval()
        self.model.load('checkpoints/WiFiModel3_0715_12_54_30.pth')
        self.model.cuda()
        self.count = 0
        self.sample_csi = []
        self.mask = numpy.zeros((1, 46, 82))

    def test(self, csi, **kwargs):
        print('start test')
        self.count = self.count + 1
        self.sample_csi.append(csi)
        if self.count == 5:
            new_csi = numpy.array(self.sample_csi)
            new_csi = numpy.concatenate(new_csi, axis = 2).transpose(2, 0, 1)
            t = torch.tensor((new_csi))
            input = Variable(t.float())
            input = input.unsqueeze(0)
            print(input)
            input = input.cuda()
            with torch.no_grad():
                score = self.model(input)

            mask_predict = score[0][:, -1, :, :]
            # sio.savemat('D:\\Project\\Person-in-WiFi\\network\\PersonWiFi\\result\\test\\reslut.mat', {'jhms': score[0].cpu().numpy().squeeze()})
            mask = Mask(mask_predict)
            my_mask = mask.get_mask()

            mask = numpy.where(my_mask == 1, 255, 0)
            print(mask.shape)
            self.mask = mask.squeeze(axis=0)
            self.sample_csi = []
            self.count = 0



        '''
        plt.figure()
        plt.subplot(111)
        plt.imshow(mask, cmap='Greys_r')
        plt.show()
        '''
        return self.mask
        #heatmap = Heatmap(score[0][:, :-1, :, :], data['JHMs'], data['video'], data['frame'])
        #my_frame = heatmap.get_image()
        #cv2.namedWindow('frame', 0)
        #print(my_frame.shape)
        #cv2.imshow('frame', my_frame)
