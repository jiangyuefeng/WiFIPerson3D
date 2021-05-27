# -*- coding: utf-8 -*-
# @Author: MapleSky
# @Date:   2021-01-22 18:31:21
# @Last Modified by:   MapleSky
# @Last Modified time: 2021-01-23 17:59:12
import json
import logging
import math
import time
from os.path import join

import cv2
import numpy as np
import scipy.io as sio
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import models
from config import opt
from data.dataset import CSIList
from utils.visualize import Visualizer, get_joint, get_box


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pose_accuary(output, box, target, alpha=50):
    video_name, frame_number = target
    target_dir = join(opt.train_data, 'alphapose', "20210417", "mu_alphapose_resize")
    jhms = output
    PCK = AverageMeter()
    for index in range(jhms.shape[0]):
        gt_dir = join(target_dir, video_name[index] + '_' + frame_number[index] + '.mat')
        gt_pose = sio.loadmat(gt_dir)['keypoints'][:, :, :-1] * 46 / 720
        gt_bbox = sio.loadmat(gt_dir)['box'][:, :-1] * 46 / 720
        output_joints, output_maxval = get_joint(jhms[index], box[index])
        #print(f'j:{output_joints.shape}')
        #print(f"gt:{gt_pose.shape}")
        for num in range(gt_pose.shape[0]):
            #print(gt_pose.shape)
            for i in range(output_joints.shape[0]):
                kp = output_joints[i]
                pck = 0
                if output_joints.shape[0] == 0:
                    pck = np.zeros(9)
                    PCK.update(pck)
                    continue
                pckPer = 0
                temPck = computePCK(kp, gt_pose[num], gt_bbox[num])
                if np.sum(temPck) > np.sum(pckPer):
                    pckPer = temPck
                pck = pck + pckPer
                PCK.update(pck)
    return PCK.avg


def computePCK(output, target, box, alpha=0.5):
    #alpha = np.tile(np.linspace(0.1, 0.9, 9), (17, 1)).transpose(1, 0)
    alpha = np.tile(np.linspace(0, 0.5, 110), (17, 1)).transpose(1, 0)
    height = box[3] - box[1]
    width = box[2] - box[0] 
    norm = math.sqrt(pow(width, 2) + pow(height, 2))
    # dist = np.linalg.norm(output, target)
    # print(output - target)
    # print(np.square(output - target))
    # print(np.sqrt(np.sum(np.square(output - target), axis=1)))
    dist = np.sqrt(np.sum(np.square(output - target), axis=1))
    dist = np.tile(dist, (110, 1))
    distNorm = dist / norm
    # print(dist * 100 / norm / alpha)
    pck_zore = np.zeros(dist.shape)
    pck_one = np.ones(dist.shape)
    pck = np.where(distNorm < alpha, pck_one, pck_zore)
    # for idx, value in enumerate(pck):
    #     if distNorm[idx] < 1:
    #         pck[idx] = 1
    # print(pck)
    # print(height, width, norm, dist)
    # if dist / norm <= alpha:
    #     pck = 1.0
    # else:
    #     pck = 0.0
    #return np.sum(pck, axis=1) / pck.shape[1]
    return pck



def mask_accuary(output, target, alpha=50):
    video_name, frame_number = target
    target_dir = join(opt.train_data, 'alphapose', "20210417", "mu_alphapose_resize")
    #print(video_name, frame_number, opt.batch_size)
    # print(target_dir)
    mIoU = AverageMeter()
    mAP = AverageMeter()
    for index in range(output.shape[0]):
        gt_dir = join(target_dir, video_name[index] + '_' + frame_number[index] + '.mat')
        # print(gt_dir)
        gt_box = sio.loadmat(gt_dir)['box'][:, :-1] * (46.0 / 720)

        #print(gt_box)
        output_bbox = get_box(output[index], 30)
        if len(output_bbox) == 0:
            continue
        IoU = 0
        for i in range(len(output_bbox)):
            aIoU = 0
            for j in range(gt_box.shape[0]):
                tempIoU = computeIoU(gt_box[j], output_bbox[i])
                if tempIoU > aIoU and tempIoU < 1 and tempIoU > 0:
                    aIoU = tempIoU
            IoU = IoU + aIoU
        IoU = IoU / float(len(output_bbox))
        # print(IoU)
        mIoU.update(IoU)
        alpha = np.linspace(0.5, 0.95, 10)
        mIoU_array = np.tile(mIoU.val, 10)
        mAP_zero = np.zeros(mIoU_array.shape)
        mAP_one = np.ones(mIoU_array.shape)
        mAP_res = np.where(mIoU_array > alpha, mAP_one, mAP_zero)
        mAP.update(mAP_res)
    # print(mAP.avg)
    return mIoU.avg, mAP.avg


def computeIoU(gt_bx, da_bx):
    xA = max(gt_bx[0], da_bx[0])
    yA = max(gt_bx[1], da_bx[1])
    xB = min(gt_bx[2], da_bx[2])
    yB = min(gt_bx[3], da_bx[3])

    interArea = (xB - xA + 1) * (yB - yA + 1)
    boxgtArea = (gt_bx[2] - gt_bx[0] + 1) * (gt_bx[3] - gt_bx[1] + 1)
    boxdaArea = (da_bx[2] - da_bx[0] + 1) * (da_bx[3] - da_bx[1] + 1)
    IoU = interArea / float(boxdaArea + boxgtArea - interArea)
    # print(IoU)
    return IoU


def train(train_dataloader, model, sm_criterion, jhms_criterion, optimizer, epoch, loss_visi, mask_score_visi, pose_score_visi, best_mask_score, best_pose_score ):
    losses = AverageMeter()
    mask_score_mIoU = AverageMeter()
    mask_score_mAP = AverageMeter()
    pose_score = AverageMeter()
    model.train()

    for i, data in enumerate(train_dataloader):
        #start_time = time.time()
        if opt.use_gpu:
            input = Variable(data['csi'].float()).cuda()
            sm_label = Variable(data['box'].float()).cuda(non_blocking=True)
            jhms_label = Variable(
                data['JHMs'].float()).cuda(non_blocking=True)
        else:
            input = Variable(data['csi'].float())
            sm_label = Variable(data['box'].float())
            jhms_label = Variable(data['JHMs'].float())

        # print("\ncsi: {}, sm_label: {}, jhms_label: {}, pafs_label: {}".format(input.shape, sm_label.shape, jhms_label.shape, pafs_target.shape))
        # print(label.shape)
        optimizer.zero_grad()

        output = model(input)
        #print(output[0].shape)
        #print(output[1].shape)
        jhms, sm = torch.split(output, (17, 2), dim=1)
        print("sm: {}, jhms: {}".format(sm.shape, jhms.shape))
        #print(sm)
        sm_weight = get_weight(sm_label, 0.5, 0.5)
        sm_loss = sm_criterion(sm,sm_label)
        #sm_loss = sm_criterion(torch.mul(sm_weight, sm),torch.mul(sm_weight, sm_label))
        jhms_weight = get_weight(jhms_label, 1, 1)
        jhms_loss = jhms_criterion(torch.mul(jhms_weight, jhms),
                                    torch.mul(jhms_weight, jhms_label))
        print("mask: {} , label: {}, sm_loss: {}, jhms_loss: {}".format(sm.shape, sm_label.shape, sm_loss, jhms_loss))
        loss =  0.1 * sm_loss + jhms_loss
        losses.update(loss.item(), input.size(0))
        start_time = time.time()
        mIoU, mAP = mask_accuary(sm, (data['video'], data['frame']))
        mask_score_mIoU.update(mIoU, input.size(0))
        mask_score_mAP.update(mAP, input.size(0))
        pose_score.update(pose_accuary(jhms, sm, (data['video'], data['frame'])), input.size(0))

        loss.backward()
        optimizer.step()
        data_time = time.time() - start_time
        print(
            f'epoch:{epoch:<8}iter:{i:<8}loss:{loss.item():<14f}time:{data_time:<8.3f}mIoU:{mask_score_mIoU.avg:<8.3f}'
        )
        logging.info(f'epoch:{epoch:<8}iter:{i:<8}loss:{loss.item():<14f}time:{data_time:<8.3f}mIoU:{mask_score_mIoU.avg:<8.3f}')
        mAP_string = 'mAP:   '
        for index in range(mask_score_mAP.avg.shape[0]):
            mAP_string = mAP_string + f'@0.{index+1}:{mask_score_mAP.avg[index]:<8.3f}'
        print(mAP_string)
        logging.info(mAP_string)
        pck_string = 'PCK:   '
        for index in range(pose_score.avg.shape[0]):
            pck_string = pck_string + f'@0.{index+1}:{pose_score.avg[index]:<8.3f}'
        print(pck_string)
        logging.info(pck_string)
        loss_visi.plot('loss', loss.item())
        mask_score_visi.plot('mAP', mask_score_mIoU.avg)
        pose_score_visi.plot('PCK', pose_score.avg)

        torch.cuda.empty_cache()
    if mask_score_mIoU.avg > best_mask_score or np.sum(pose_score.avg) > np.sum(best_pose_score) :
        model.save()

def validate(val_dataloader, model, sm_criterion, jhms_criterion, epoch):
    mask_score_mIoU = AverageMeter()
    mask_score_mAP = AverageMeter()
    pose_score = AverageMeter()

    model.eval()
    for i, data in enumerate(val_dataloader):
        start_time = time.time()
        input = Variable(data['csi'].float())
        if opt.use_gpu:
            input = input.cuda()
        with torch.no_grad():
            output = model(input)

        jhms, sm = torch.split(output, (17, 2), dim=1)
        mIoU, mAP = mask_accuary(sm, (data['video'], data['frame']))
        mask_score_mIoU.update(mIoU, input.size(0))
        mask_score_mAP.update(mAP, input.size(0))
        pose_score.update(pose_accuary(jhms, sm, (data['video'], data['frame'])), input.size(0))
        data_time = time.time() - start_time
        print(f'validata: epoch:{epoch:<8}iter:{i:<8}time:{data_time:<8.3f}mIou:{mask_score_mIoU.avg:<8.3f}')
        logging.info(f'validata: epoch:{epoch:<8}iter:{i:<8}time:{data_time:<8.3f}mIoU:{mask_score_mIoU.avg:<8.3f}')
        mAP_string = 'mAP:   '
        for index in range(mask_score_mAP.avg.shape[0]):
            mAP_string = mAP_string + f'@0.{index+1}:{mask_score_mAP.avg[index]:<8.3f}'
        print(mAP_string)
        logging.info(mAP_string)
        pck_string = 'PCK:   '
        for index in range(pose_score.avg.shape[0]):
            pck_string = pck_string + f'@0.{index+1}:{pose_score.avg[index]:<8.3f}'
        print(pck_string)
        logging.info(pck_string)
        torch.cuda.empty_cache()
    return mask_score_mIoU.avg, pose_score.avg


def get_weight(gt, k, b):
    weight_positive = torch.ones(gt.shape).cuda()
    weight_negative = -torch.ones(gt.shape).cuda()
    weight = k * gt + b * torch.where(gt >= 0, weight_positive,
                                      weight_negative)
    return weight


def main(**kwargs):
    print('start')
    opt.parse(kwargs)
    # step1 loading model
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # step2 loading data
    datasets = CSIList(opt.train_data)
    print(len(datasets))
    train_size = int(0.8 * len(datasets))
    val_size = len(datasets) - train_size
    train_data, val_data = torch.utils.data.random_split(datasets, [train_size, val_size])

    # train_data = CSIList(opt.train_data)
    print(len(train_data))
    # val_data = CSIList(opt.train_data, train=False)
    train_dataloader = DataLoader(
        train_data,
        opt.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_dataloader = DataLoader(
        val_data,
        opt.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # step3 loss function and optim
    sm_criterion = torch.nn.BCEWithLogitsLoss()
    sm_criterion.cuda()
    jhms_criterion = torch.nn.MSELoss()
    jhms_criterion.cuda()

    lr = opt.lr
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr,
        betas=(0.9, 0.999),
        #weight_decay=opt.weight_decay
    )
    cudnn.benchmark = True
    best_mask_score = 0
    best_pose_score = 0

    # step4 statistic function
    loss_visi = Visualizer(env='main')
    mask_score_visi = Visualizer(env='main')
    pose_score_visi = Visualizer(env='main')
    #StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    # step5 record result
    logging.basicConfig(filename='result.txt', level=logging.INFO, filemode='w')

    for epoch in range(opt.max_epoch):
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        train(train_dataloader, model, sm_criterion, jhms_criterion, optimizer, epoch, loss_visi, mask_score_visi, pose_score_visi,best_mask_score, best_pose_score)
        mask_score, pose_score = validate(val_dataloader, model, sm_criterion, jhms_criterion, epoch)
        if mask_score > best_mask_score:
            best_mask_score = mask_score
        if np.sum(pose_score) > np.sum(best_pose_score):
            best_pose_score = pose_score
        #StepLR.step()


if __name__ == "__main__":
    main()
