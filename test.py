from config import opt
import models
import torch
import os
import matplotlib.pyplot as plt
from data.dataset import CSIList
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.joint import Joint
from utils.heatmap import Heatmap
from utils.mask import Mask
import scipy.io as sio
import main
import numpy as np
from main import AverageMeter

def getPlot(pck):
    x = np.linspace(0, 0.5, 110)
    plt.figure()
    label1 = ['Nose','LEye','REye','LEar','REar',
              'LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist',
              'LHip','RHip','LKnee','RKnee','LAnkle','RAnkle']
    for i in range(0,5):
        y = pck[:,i]
        plt.plot(x, y, label=label1[i])
    
    plt.title('')
    plt.legend()
    plt.xlabel('Normalized Distance Error')
    plt.ylabel('PCK')
    plt.xlim(0,0.55)
    plt.ylim(0,0.8)
    plt.figure()
    for i in range(5,11):
        y = pck[:,i]
        plt.plot(x, y, label=label1[i])
    
    plt.title('')
    plt.legend()
    plt.xlabel('Normalized Distance Error')
    plt.ylabel('PCK')
    plt.xlim(0,0.55)
    plt.ylim(0,0.8)

    plt.figure()
    for i in range(11,17):
        y = pck[:,i]
        plt.plot(x, y, label=label1[i])
    
    plt.title('')
    plt.legend()
    plt.xlabel('Normalized Distance Error')
    plt.ylabel('PCK')
    plt.xlim(0,0.55)
    plt.ylim(0,0.8)
    
    plt.show()

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
    print(len(test_data))
    test_dataloader = DataLoader(
        test_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
    )
    kpts = []
    val = 0 
    p_sum = 0
    count = 0
    avg = 0
    PCK = AverageMeter()
    mIoU = AverageMeter()
    mAP = AverageMeter()
    for index, data in enumerate(test_dataloader):
        #print(data['SM'])
        input = Variable(data['csi'].float())
        if opt.use_gpu:
            input = input.cuda()
        with torch.no_grad():
            score = model(input)
        jhm, mask_predict = torch.split(score, (17, 2), dim=1)
        for i in range(opt.batch_size):
            pic = index * 32 + i
            if True:
                print(index)
                # sio.savemat('D:\\Project\\Person-in-WiFi\\network\\PersonWiFi\\result\\test\\reslut.mat', {'jhms': score[0].cpu().numpy().squeeze()})
                joint1 = Joint(jhm[i], mask_predict[i], data['video'][i], data['frame'][i], pad=True)
                #joint2 = Joint(data['JHMs'][i], data['box'][i], data['video'][i], data['frame'][i], pad=True)
                #joint1.save_img()
                joint1.sh_joint()
                #joint2.sh_joint()
                heatmap = Heatmap(jhm[i], data['JHMs'][i], data['video'][i], data['frame'][i])
                heatmap.show_all()
                heatmap.show_sp()
                mask = Mask(mask_predict[i], data['box'][i])
                mask.show()
                plt.show() 
        '''      
        pck = main.pose_accuary(jhm, mask_predict, (data['video'], data['frame']))
        miou, ap = main.mask_accuary(mask_predict, (data['video'], data['frame']))
        PCK.update(pck)
        mAP.update(ap)
        mIoU.update(miou)      
    print(mIoU.avg)
    getPlot(PCK.avg)
    '''

    '''
    preds, maxvals = main.get_joint(jhm[i][:-1, :, :])
    kpts.append(preds)
    name = f'{opt.testdata}/np/20210116/{video_name}.npz'
    kpts = np.array(kpts).astype(np.float32)
    print('kpts npz save in ', name)
    np.savez_compressed(name, kpts=kpts)
    '''
    '''
    mask = Mask(mask_predict[i], data['box'][i])
    print(data['box'][i][0])
    mask.show()
    heatmap = Heatmap(jhm[i], data['JHMs'][i], data['video'][i], data['frame'][i])
    heatmap.show_all()
    heatmap.show_sp()
    Joint(data['JHMs'][i], data['video'][i], data['frame'][i], pad=False)
    Joint(jhm[i], data['video'][i], data['frame'][i], pad=True)
    plt.show()
    '''


if __name__ == "__main__":
    test()
