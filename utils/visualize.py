import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib
import visdom
import math



def get_dist(a, b):
    dis = np.sqrt(math.pow(a[1]-b[1], 2)+math.pow(a[0]-b[0], 2))
    return dis

def get_joint(batch_heatmaps, box):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    batch_heatmaps = batch_heatmaps.cpu().detach().numpy()
    
    bbox = get_box(box, 30)
    final_preds = []
    final_maxvals = []
    for num in range(bbox.shape[0]):
        positive = np.zeros(batch_heatmaps[0].shape)
        positive[(bbox[num][1]):(bbox[num][3]), (bbox[num][0]):(bbox[num][2])] = 1
        # heatmap mul box
        heatmaps = batch_heatmaps.copy()
        for i in range(batch_heatmaps.shape[0]):
            heatmaps[i] = np.multiply(heatmaps[i], positive)
        num_joints = batch_heatmaps.shape[0]
        width = heatmaps.shape[2]
        heatmaps_reshaped = heatmaps.reshape((num_joints, -1))
        #print(heatmaps_reshaped.shape)
        idx = np.argmax(heatmaps_reshaped, 1)
        #print(idx)
        maxvals = np.max(heatmaps_reshaped, 1)

        maxvals = maxvals.reshape((num_joints, 1))
        idx = idx.reshape((num_joints, 1))

        preds = np.tile(idx, (1, 2)).astype(np.float32)

        preds[:, 0] = (preds[:, 0]) % width
        preds[:, 1] = np.floor((preds[:, 1]) / width)
        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 2))
        pred_mask = pred_mask.astype(np.float32)
        preds *= pred_mask
        # preds= re_get_joint(preds)
        final_preds.append(preds)
        final_maxvals.append(maxvals)
    final_preds = np.array(final_preds)
    final_maxvals = np.array(final_maxvals)
    #print(final_preds.shape)
    return final_preds, final_maxvals

def get_box(data, maxarea):
    data = data.cpu().detach().numpy()
    sort_bbox = None
    person_num = 0
    for i in range(data.shape[0]):
        bbox = []
        delete = False
        mask = data[i].copy()
        positive = np.ones(mask.shape) * 255
        negative = np.zeros(mask.shape)
        threshold = np.max(mask) * 0.34
        mask = np.where(mask > threshold, positive, negative).astype(np.uint8)
        # print(mask.shape)
        if cv2.__version__.split('.')[0] == "4":
            contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            img, contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x1, y1, w, h = cv2.boundingRect(c)
            x1 = x1 - (2*i +1)
            y1 = y1 - (2*i +1)
            x2 = x1 + w + (2*i +2)
            y2 = y1 + h + (2*i +2)
            area = cv2.contourArea(c)
            if area > maxarea:
                bbox.append([x1, y1, x2, y2])
        sort_bbox = np.array(bbox)
        if(sort_bbox.shape[0] != 0):
            sort_bbox = sort_bbox[np.lexsort(sort_bbox[:,::-1].T)]
            for i in range(1,sort_bbox.shape[0]):
                if(sort_bbox[i-1][2]>sort_bbox[i][2]):
                    delete = True
        if sort_bbox.shape[0] > person_num and delete == False:
            person_num = sort_bbox.shape[0]
            final_bbox = sort_bbox
    return final_bbox

def get_joint_dist(batch_heatmaps, box):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    batch_heatmaps = batch_heatmaps.cpu().detach().numpy()
    bbox = np.array(get_box(box,30))
    num_joints = batch_heatmaps.shape[0]
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 3, 'batch_images should be 3-ndim'
    preds_all = np.zeros((0, 17, 2))
    maxVal_all = np.zeros((0, 17, 1))
    for num in range(bbox.shape[0]):
        mask = np.zeros(box[0].shape)
        mask[(bbox[num][1]):(bbox[num][3]) + 1, (bbox[num][0]):(bbox[num][2]) + 1] = 1
        
        heatmaps = batch_heatmaps.copy()
        # heatmap mul box
        width = heatmaps.shape[2]
        for i in range(num_joints):
            heatmaps[i] = np.multiply(heatmaps[i], mask)
        mapSmooth = cv2.GaussianBlur(heatmaps,(3,3),0,0)
        threshold = np.max(heatmaps) * 0.3
        mapMask = np.uint8(mapSmooth > threshold)
        keypoints = []
        for i in range(mapMask.shape[0]):
            j_keypoints = []
            if cv2.__version__.split('.')[0] == "4":
                contours, hier = cv2.findContours(mapMask[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            else:
                img, contours, hier = cv2.findContours(mapMask[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                blobMask = np.zeros(mapMask[i].shape)
                blobMask = cv2.fillConvexPoly(blobMask,cnt,1)
                maskProbMap = mapSmooth*blobMask
                _,maxVal,_,maxLoc = cv2.minMaxLoc(maskProbMap[i])
                j_keypoints.append({
                    "maxLoc":maxLoc,
                    "score":batch_heatmaps[i][maxLoc[1],maxLoc[0]],
                    "num":i
                })
            keypoints.append(j_keypoints) 
        mapIdx = [1,3,4,1,2,7,8,9,10,7,8,13,14,15,16,13,14]

        preds = []
        maxVal = []
        for i in range(num_joints):
            if(len(keypoints[i])==1):
                preds.append(keypoints[i][0]["maxLoc"])
                maxVal.append(keypoints[i][0]["score"])
            else:
                min_dis = 10000
                min_preds = (0, 0)
                min_score = (0)
                for j_n in range(len(keypoints[i])):
                    mapid = mapIdx[i]
                    if len(keypoints[mapid]) == 0:
                        kp_score = keypoints[i][j_n]["score"]
                        if min_score < kp_score:
                            min_preds = keypoints[i][j_n]["maxLoc"]
                            min_score = keypoints[i][j_n]["score"]
                    else:
                        dis = get_dist(keypoints[i][j_n]["maxLoc"],keypoints[mapid][0]["maxLoc"])
                        if dis < min_dis:
                            min_dis = dis
                            min_preds = keypoints[i][j_n]["maxLoc"]
                            min_score = keypoints[i][j_n]["score"]
                preds.append(min_preds)
                maxVal.append(min_score)
        preds = np.expand_dims(np.array(preds),axis=0)
        maxVal= np.expand_dims(np.expand_dims(np.array(maxVal),axis=1),axis=0)
        preds_all = np.vstack((preds_all, preds))
        maxVal_all = np.vstack((maxVal_all,maxVal))
    #print(preds_all.shape)   
    return preds_all, maxVal_all

def re_get_joint(preds):

    # print(f'heatmap:{batch_heatmaps[0]}')
    part_list = [(9, 7),(10, 8),(15, 13),(16, 14)]
    root_list = [(5, 7),(6, 8),(11, 13),(12, 14)]
    for i, (start_p, end_p) in enumerate(part_list):
        part = get_dist(preds[start_p],preds[end_p])
        root = get_dist(preds[root_list[i][0]], preds[root_list[i][1]])
        if(part > 1.5 * root):
            preds[start_p] = 2 * preds[end_p] - preds[root_list[i][0]]
    return preds

def display_mask(mask, gt):
    '''
       mask      predict mask          (1,46,82)
       gt        groudtruth mask       (1,46,82)    
    '''
    #print(mask.shape)
    #print(gt.shape)
    mask = np.where(mask == 1, 255, 0)
    plt.figure()
    plt.subplot(121)
    plt.imshow(mask, cmap='Greys_r')

    gt = np.where(gt == 1, 255, 0)
    gt = gt.squeeze()
    plt.subplot(122)
    plt.imshow(gt, cmap='Greys_r')



def display_heatmap(heatmap, gt, frame):

    heatmap = heatmap.squeeze()
    gt = gt.squeeze()
    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1)
    demo_locatable_axes_easy(ax, heatmap, frame)

    ax_gt = fig.add_subplot(1, 2, 2)
    demo_locatable_axes_easy(ax_gt, gt, frame)


def demo_locatable_axes_easy(ax, join_image, label=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)

    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)

    ax.imshow(label)

    join_image = np.average(np.array(join_image), axis=0)
    join_image = cv2.resize(join_image, (1280, 720))
    im = ax.imshow(join_image, alpha=1)
    # for i in range(image.shape[0]):
    #     im = ax.imshow(image[i, :, :], alpha=0.5)

    plt.colorbar(im, cax=ax_cb)
    ax_cb.yaxis.tick_right()
    ax_cb.yaxis.set_tick_params(labelright=True)


def display_joints(joints, image):
    colors = [[255, 0, 0], [255, 51, 0], [255, 102, 0], [255, 153, 0], [255, 204, 0], [255, 255, 0], \
              [204, 255, 0], [153, 255, 0], [102, 255, 0], [51, 255, 0], [0, 255, 0], \
              [0, 255, 51], [0, 255, 102], [0, 255, 153], [0, 255, 204], [0, 255, 255], \
              [0, 204, 255]]
    cmap = matplotlib.cm.get_cmap('hsv')
    canvas = image

    for i in range(17):
        rgba = np.array(cmap(1 - i / 18. - 1. / 36))
        rgba[0:3] *= 255
        for j in range(len(joints[i])):
            if joints[i][j][2] > 0.2:
                cv2.circle(
                    canvas,
                    (int(joints[i][j][0] * 15.6), int(joints[i][j][1] * 15.6)),
                    4,
                    colors[i],
                    thickness=-1)

    to_plot = cv2.addWeighted(image, 0.3, canvas, 0.7, 0)
    plt.imshow(to_plot[:, :, [2, 1, 0]])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(12, 12)
    plt.show()


def display_pafs(X, Y, U, V, image):
    plt.figure()
    plt.imshow(image, alpha=0.5)
    s = 10
    if X.ndim == 3:
        for i in range(X.shape[0]):
            Q = plt.quiver(X[i][::s, ::s],
                           Y[i][::s, ::s],
                           U[i][::s, ::s],
                           V[i][::s, ::s],
                           scale=50,
                           headaxislength=4,
                           alpha=.5,
                           width=0.001,
                           color='r')
    else:
        Q = plt.quiver(X[::s, ::s],
                       Y[::s, ::s],
                       U[::s, ::s],
                       V[::s, ::s],
                       scale=50,
                       headaxislength=4,
                       alpha=.5,
                       width=0.001,
                       color='r')


class Visualizer(object):
    def __init__(self, env='defualt', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}
        self.log_text = ''

    def reinit(self, env='defualt', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot(self, name, y, **kwargs):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]),
                      X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs)
        self.index[name] = x + 1
