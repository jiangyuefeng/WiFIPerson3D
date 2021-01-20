from config import opt
import torch
import models
from data.dataset import CSIList
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
from utils.visualize import Visualizer
from os.path import join
import scipy.io as sio
import numpy as np
import cv2
import math
import torch.backends.cudnn as cudnn
import logging


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


def pose_accuary(output, target, alpha=50):
    video_name, frame_number = target
    target_dir = join(opt.train_data, 'allignedPose')
    jhms, pafs = output
    PCK = AverageMeter()
    for index in range(jhms.shape[0]):
        gt_dir = join(target_dir, video_name[index] + '_' + frame_number[index] + '.mat')
        gt_pose = sio.loadmat(gt_dir)['openpose_array'][:, :, :-1] * 46 / 720
        gt_bbox = sio.loadmat(gt_dir)['boxes'][:, :-1] * 46 / 720
        # print(gt_pose.shape)
        output_joints = get_joint(jhms[index], pafs[index])
        pck = 0
        if output_joints.shape[0] == 0:
            pck = np.zeros(9)
            PCK.update(pck)
            continue
        for i in range(output_joints.shape[0]):
            pckPer = 0
            for j in range(gt_pose.shape[0]):
                temPck = computePCK(output_joints[i], gt_pose[j], gt_bbox[j])
                if np.sum(temPck) > np.sum(pckPer):
                    pckPer = temPck
            pck = pck + pckPer
        pck = pck / float(output_joints.shape[0])
        PCK.update(pck)
    # print(PCK.avg)
    return PCK.avg


def computePCK(output, target, box, alpha=0.5):
    alpha = np.tile(np.linspace(0.1, 0.9, 9), (25, 1)).transpose(1, 0)
    height = box[3] - box[1]
    width = box[2] - box[0]
    norm = math.sqrt(pow(width, 2) + pow(height, 2))
    # dist = np.linalg.norm(output, target)
    # print(output - target)
    # print(np.square(output - target))
    # print(np.sqrt(np.sum(np.square(output - target), axis=1)))
    dist = np.sqrt(np.sum(np.square(output - target), axis=1))
    dist = np.tile(dist, (9, 1))
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
    return np.sum(pck, axis=1) / pck.shape[1]


def get_joint(jhms, pafs):
    from scipy.ndimage.filters import gaussian_filter
    jhms = jhms.cpu().detach().numpy().transpose(1, 2, 0)
    pafs = pafs.cpu().detach().numpy().transpose(1, 2, 0)
    # print(jhms.shape, pafs.shape)
    all_peaks = []
    peak_counter = 0
    for part in range(25):
        map_ori = jhms[:, :, part]
        maps = gaussian_filter(map_ori, sigma=0.3)

        map_left = np.zeros(maps.shape)
        map_left[1:, :] = maps[:-1, :]
        map_right = np.zeros(maps.shape)
        map_right[:-1, :] = maps[1:, :]
        map_up = np.zeros(maps.shape)
        map_up[:, 1:] = maps[:, :-1]
        map_down = np.zeros(maps.shape)
        map_down[:, :-1] = maps[:, 1:]

        peaks_binary = np.logical_and.reduce((maps >= map_left, maps >= map_right, maps >= map_up, maps >= map_down, maps > 0.1))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))
        peaks_with_score = [x[:2] + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]
        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)
    # print(all_peaks[-1][-1][-1])
    if all_peaks[-1][-1][-1] > 500:
        return np.array([])

    limbSeq = [[2, 9], [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8],
               [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
               [2, 1], [1, 16], [16, 18], [1, 17], [17, 19], [3, 18],
               [6, 19], [15, 20], [20, 21], [15, 22], [12, 23], [23, 24],
               [12, 25]]
    mapIdx = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13],
              [14, 15], [16, 17], [18, 19], [20, 21], [22, 23], [24, 25],
              [26, 27], [28, 29], [30, 31], [32, 33], [34, 35], [36, 37],
              [38, 39], [40, 41], [42, 43], [44, 45], [46, 47], [48, 49],
              [50, 51]]

    connection_all = []
    special_k = []
    mid_num = 5
    for k in range(len(mapIdx)):
        score_mid = pafs[:, :, [x for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if nA != 0 and nB != 0:
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(pow(vec[0], 2) + pow(vec[1], 2))
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                        np.linspace(candA[i][1], candB[j][1], num=mid_num)))
                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] for I in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] for I in range(len(startend))])
                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * 46 / norm - 1, 0)
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
            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])
    # print(len(connection_all))

    subset = -1 * np.ones((0, 27))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1
                if found == 1:
                    j = subset_idx[0]
                    if(subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    # print("found = 2")
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 24:
                    row = -1 * np.ones(27)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)
    # print(len(subset), len(subset[1]))

    joint = -1 * np.zeros((len(subset), 25, 2))
    for personid in range(len(subset)):
        for squeid in range(25):
            if subset[personid][squeid] != -1:
                for index in range(len(all_peaks[squeid])):
                    if all_peaks[squeid][index][3] == subset[personid][squeid]:
                        joint[personid, squeid, :] = np.array([all_peaks[squeid][index][0:2]])
    # print(joint.shape)
    return joint


def mask_accuary(output, target, alpha=50):
    video_name, frame_number = target
    # print(video_name, frame_number, opt.batch_size)
    target_dir = join(opt.train_data, 'mask')
    # print(target_dir)
    mIoU = AverageMeter()
    mAP = AverageMeter()
    for index in range(output.shape[0]):
        gt_dir = join(target_dir, video_name[index] + '_' + frame_number[index] + '.mat')
        # print(gt_dir)
        gt_box = sio.loadmat(gt_dir)['boxes'][:, :-1] * (46.0 / 720)
        # print(gt_box)
        output_bbox = get_box(output[index])
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
        alpha = np.linspace(0.1, 0.9, 9)
        mIoU_array = np.tile(mIoU.val, 9)
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


def get_box(data):
    mask = data.cpu().detach().numpy().squeeze()
    positive = np.ones(mask.shape) * 255
    negative = np.zeros(mask.shape)
    threshold = np.max(mask) * 0.5
    mask = np.where(mask > threshold, positive, negative).astype(np.uint8)
    # print(mask.shape)
    if cv2.__version__.split('.')[0] == "4":
        contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        img, contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bbox = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area > 377 or area < 3000:
            bbox.append([x, y, x + w, y + h])
    # print(bbox)
    return bbox


def train(train_dataloader, model, sm_criterion, jhms_criterion, pafs_criterion, optimizer, epoch, loss_visi, mask_score_visi, pose_score_visi):
    losses = AverageMeter()
    mask_score_mIoU = AverageMeter()
    mask_score_mAP = AverageMeter()
    pose_score = AverageMeter()

    model.train()

    for i, data in enumerate(train_dataloader):
        start_time = time.time()
        if opt.use_gpu:
            input = Variable(data['csi'].float()).cuda()
            sm_label = Variable(data['SM'].float()).cuda(non_blocking=True)
            jhms_label = Variable(
                data['JHMs'].float()).cuda(non_blocking=True)
            pafs_label = Variable(
                data['PAFs'].float()).cuda(non_blocking=True)
        else:
            input = Variable(data['csi'].float())
            sm_label = Variable(data['SM'].float())
            jhms_label = Variable(data['JHMs'].float())
            pafs_label = Variable(data['PAFs'].float())

        # print("\ncsi: {}, sm_label: {}, jhms_label: {}, pafs_label: {}".format(input.shape, sm_label.shape, jhms_label.shape, pafs_target.shape))
        # print(label.shape)
        optimizer.zero_grad()

        output = model(input)

        #print(output[0].shape)
        #print(output[1].shape)
        jhms, sm = torch.split(output[0], (17, 1), dim=1)
        pafs = output[1]
        print("sm: {}, jhms: {}, pafs: {}".format(sm.shape, jhms.shape, pafs.shape))
        print(sm)
        sm_loss = sm_criterion(torch.split(output[0], (17, 1), dim=1)[1], sm_label)
        jhms_weight = get_weight(jhms_label, 1, 1)
        jhms_loss = jhms_criterion(torch.mul(jhms_weight, torch.split(output[0], (17, 1), dim=1)[0]),
                                    torch.mul(jhms_weight, jhms_label))
        pafs_weight = get_weight(pafs_label, 1, 0.3)
        pafs_loss = pafs_criterion(torch.mul(pafs_weight, output[1]),
                                    torch.mul(pafs_weight, pafs_label))
        # print("mask: {} , label: {}, sm_loss: {}, jhms_loss: {}, pafs_loss: {}".format(sm.shape, sm_label.shape, sm_loss, jhms_loss, pafs_loss))
        loss = 0.1 * sm_loss + jhms_loss + pafs_loss
        losses.update(loss.item(), input.size(0))
        mIoU, mAP = mask_accuary(torch.split(output[0], (17, 1), dim=1)[1], (data['video'], data['frame']))
        mask_score_mIoU.update(mIoU, input.size(0))
        mask_score_mAP.update(mAP, input.size(0))
        pose_score.update(pose_accuary((jhms, pafs), (data['video'], data['frame'])), input.size(0))

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
        # pose_score_visi.plot('PCK', pose_score.avg)

        torch.cuda.empty_cache()
    # if mask_score.avg > best_mask_score:
    #     best_mask_score = mask_score.avg
        # best_pose_score = pose_score.avg
        # model.save()
    model.save()


def validate(val_dataloader, model, sm_criterion, jhms_criterion, pafs_criterion, epoch):
    mask_score_mIoU = AverageMeter()
    mask_score_mAP = AverageMeter()
    pose_score = AverageMeter()

    model.eval()
    start_time = time.time()

    for i, data in enumerate(val_dataloader):
        input = Variable(data['csi'].float())
        if opt.use_gpu:
            input = input.cuda()
        with torch.no_grad():
            output = model(input)

        jhms, sm = torch.split(output[0], (25, 1), dim=1)
        pafs = output[1]
        mIoU, mAP = mask_accuary(torch.split(output[0], (25, 1), dim=1)[1], (data['video'], data['frame']))
        mask_score_mIoU.update(mIoU, input.size(0))
        mask_score_mAP.update(mAP, input.size(0))
        pose_score.update(pose_accuary((jhms, pafs), (data['video'], data['frame'])), input.size(0))
    data_time = time.time() - start_time
    print(f'validata: epoch:{epoch:<8}time:{data_time:<8.3f}mIou:{mask_score_mIoU.avg:<8.3f}')
    logging.info(f'validata: epoch:{epoch:<8}time:{data_time:<8.3f}mIoU:{mask_score_mIoU.avg:<8.3f}')
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
    train_size = int(0.7 * len(datasets))
    val_size = len(datasets) - train_size
    train_data, val_data = torch.utils.data.random_split(datasets, [train_size, val_size])

    # train_data = CSIList(opt.train_data)
    print(len(train_data))
    # val_data = CSIList(opt.train_data, train=False)
    train_dataloader = DataLoader(
        train_data,
        opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
    )
    val_dataloader = DataLoader(
        val_data,
        opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers
    )

    # step3 loss function and optim
    sm_criterion = torch.nn.BCEWithLogitsLoss()
    sm_criterion.cuda()
    jhms_criterion = torch.nn.MSELoss()
    jhms_criterion.cuda()
    pafs_criterion = torch.nn.MSELoss()
    pafs_criterion.cuda()

    lr = opt.lr
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr,
        betas=(0.9, 0.999),
        # weight_decay=opt.weight_decay
    )
    cudnn.benchmark = True
    best_mask_score = 0
    best_pose_score = 0

    # step4 statistic function
    loss_visi = Visualizer(env='main')
    mask_score_visi = Visualizer(env='main')
    pose_score_visi = Visualizer(env='main')

    # step5 record result
    logging.basicConfig(filename='result.txt', level=logging.INFO, filemode='w')

    for epoch in range(opt.max_epoch):
        train(train_dataloader, model, sm_criterion, jhms_criterion, pafs_criterion, optimizer, epoch, loss_visi, mask_score_visi, pose_score_visi)
        mask_score, pose_score = validate(val_dataloader, model, sm_criterion, jhms_criterion, pafs_criterion, epoch)
        if mask_score > best_mask_score:
            best_mask_score = mask_score
        if np.sum(pose_score) > np.sum(best_pose_score):
            best_pose_score = pose_score


if __name__ == "__main__":
    main()
