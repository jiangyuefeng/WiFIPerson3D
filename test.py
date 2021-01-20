from config_test import opt
import models
import torch
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from data.dataset import CSIList
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.mask import Mask
import cv2
from utils.heatmap import Heatmap
from utils.vectormap import Vectormap
from utils.joint import Joint
import scipy.io as sio
import numpy


def test(**kwargs):
    print('start test')
    opt.parse(kwargs)
    # print(opt)

    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    test_data = CSIList(opt.test_data)
    test_dataloader = DataLoader(
        test_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
    )
    for index, data in enumerate(test_dataloader):
        input = Variable(data['csi'].float())
        if opt.use_gpu:
            input = input.cuda()
        with torch.no_grad():
            score = model(input)

        mask_predict = score[0][:, -1, :, :]
        # sio.savemat('D:\\Project\\Person-in-WiFi\\network\\PersonWiFi\\result\\test\\reslut.mat', {'jhms': score[0].cpu().numpy().squeeze()})
        mask = Mask(mask_predict)

        my_mask = mask.get_mask()

        mask = numpy.where(my_mask == 1, 255, 0)
        print(mask.shape)
        mask = mask.squeeze(axis=0)
        #heatmap = Heatmap(score[0][:, :-1, :, :], data['JHMs'], data['video'], data['frame'])
        #my_frame = heatmap.get_image()
        #cv2.namedWindow('frame', 0)
        #print(my_frame.shape)
        #cv2.imshow('frame', my_frame)
        '''
        plt.figure()
        plt.subplot(111)
        plt.imshow(mask, cmap='Greys_r')
        plt.show()
        '''
        mask = mask.astype(numpy.uint8)
        cv2.namedWindow('mask', 0)
        cv2.imshow('mask',mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test()
