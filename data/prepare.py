from os.path import join
import json
import math
import os
import cv2
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

def get_bbox(ori_size,box,target_size):
    width, height = target_size
    bbox = np.zeros((2, height, width), dtype=np.float32)
    for person in range(box.shape[0]):
        for idx, point in enumerate(box[person]):
            if point <= 0:
                continue
            bbox = put_box(bbox, box[person])
    bbox = bbox.transpose(1, 2, 0)
    #print(bbox.shape)
    return bbox.astype(np.float16)

def put_box(bbox, box):
    box = box.astype(np.int16)
    box_list = [1,3,4,5]
    for i in range(len(box_list)):
        y1 = box[1] + box_list[i]
        y2 = box[3] - box_list[i]
        x1 = box[0] + box_list[i]
        x2 = box[2] - box_list[i]
        if(y1<y2 or x1<x2):
        ## fast - vectorize
            bbox[i, y1:y2, x1:x2] = 1
    return bbox

def get_heatmap(ori_size, pose, target_size):
    width, height = target_size
    heatmap = np.zeros((pose.shape[1] + 1, height, width), dtype=np.float32)
    # print(pose[0][0])
    for person in range(pose.shape[0]):
        for idx, point in enumerate(pose[person]):
            if point[0] <= 0 or point[1] <= 0:
                continue
            if point[2] <= 0.05:
                continue
            point = point[0:2] * 0.064
            heatmap = put_heatmap(heatmap, idx, point[0:2], 1.0)
    # print(heatmap[24, 492:540, 779:828])
    # sio.savemat("D:\Project\Person-in-WiFi\dataset\mask_resize\\" + "heatmap.mat", {'heatmap': np.array(heatmap)})
    heatmap = heatmap.transpose((1, 2, 0))
    heatmap[:, :, -1] = np.clip(1 - np.amax(heatmap, axis=2), 0.0, 1.0)

    # if target_size:
    #     mapholder = []
    #     for i in range(0, pose.shape[1] + 1):
    #         a = cv2.resize(np.array(heatmap[:, :, i]), target_size, interpolation=cv2.INTER_CUBIC)
    #         mapholder.append(a)
    #     mapholder = np.array(mapholder)
    #     heatmap = mapholder.transpose(1, 2, 0)
    # print(np.sum(heatmap[:, :, :-1]))    
    return heatmap.astype(np.float16)


def put_heatmap(heatmap, plane_idx, center, sigma):
    center_x, center_y = center
    # print(center_x, center_y)
    _, height, width = heatmap.shape[:3]
    th = 4.6052
    delta = math.sqrt(th * 2)

    x0 = int(max(0, center_x - delta * sigma + 0.5))
    y0 = int(max(0, center_y - delta * sigma + 0.5))

    x1 = int(min(width - 1, center_x + delta * sigma + 0.5))
    y1 = int(min(height - 1, center_y + delta * sigma + 0.5))

    exp_factor = 1 / 2.0 / sigma / sigma

    ## fast - vectorize
    arr_heatmap = heatmap[plane_idx, y0:y1 + 1, x0:x1 + 1]
    y_vec = (np.arange(y0, y1 + 1) - center_y)**2  # y1 included
    x_vec = (np.arange(x0, x1 + 1) - center_x)**2
    xv, yv = np.meshgrid(x_vec, y_vec)
    arr_sum = exp_factor * (xv + yv)
    arr_exp = np.exp(-arr_sum)
    arr_exp[arr_sum > th] = 0
    # print(arr_sum)
    # print(arr_exp)
    # print(np.maximum(arr_heatmap, arr_exp))
    # print(plane_idx, y0, y1, x0, x1)
    heatmap[plane_idx, y0:y1 + 1, x0:x1 + 1] = np.maximum(arr_heatmap, arr_exp)
    # print(np.sum(heatmap[plane_idx, y0:y1 + 1, x0:x1 + 1] != 0))
    return heatmap


if __name__ == "__main__":
    root_dir = "/home/public/b509/code/dataset"
    alphapose_dir = join(root_dir, "alphapose", "20210417")
    for frame_index in range(40,41):
        file_name = "2021-4-18_p" + str(frame_index) + "_video"
        json_dir = join(alphapose_dir, "AlphaPose_out", file_name,
                        "alphapose-results.json")
        # print(json_dir)
        with open(json_dir) as f:
            results = json.load(f)
            for i in range(len(results)):
                keypoints = np.array(results[i]["keypoints"])
                box = np.array(results[i]['box'])
                bbox = box[:, :-1] * (46.0 / 720)
                heatmap = get_heatmap((1280, 720), keypoints, (82, 46))
                bbox = get_bbox((1280, 720), bbox, (82, 46))
                resize_name = file_name + "_" + str(i) + '.mat'
                '''
                mask = bbox[:,:,1]
                mask = np.where(mask == 1, 255, 0)
                
                plt.figure()
                plt.imshow(mask, cmap='Greys_r')
                plt.show()
                '''
                sio.savemat(join(alphapose_dir, 'mu_alphapose_resize',resize_name), {
                    'heatmap':heatmap,
                    'keypoints': keypoints,
                    'bbox':bbox,
                    'box': box,
                })
    '''
    # jhms and pafs
    for root, dirs, files in os.walk(pose_dir):
        print(root)
        for name in files:
            if os.path.splitext(name)[-1] == '.json':
                print(name)
                json_dir = join(root, 'alphapose_results.json')
                with open(json_dir) as f:
                    result_resize = []
                    # print(f)
                    sample_json = json.load(f)
                    for sample in sample_json:
                        result = {}
                        JHM = sample['keypoints']
                        box = sample['box']
                        image_id = sample['image_id']
                        num = image_id.split('.')[0]
                        #print(image_id)
                        JHM = np.array(JHM)
                        #print(JHM.shape)
                        # 转变float64 sample_JHMs torch.size(17,46,82) sample_PAFs torch.size(36,46,82)
                        sample_JHMs = get_heatmap(JHM,(82, 46)).tolist()
                        sample_PAFs = get_vectormap(JHM, (82, 46)).tolist()
                        
                        result["image_id"] = image_id
                        result["JHM"] = sample_JHMs
                        result["PAF"] = sample_PAFs
                        result["box"] = box
                        result_resize.append(result)
                        
                        sio.savemat(join(root_dir, "res", 'pose_resize',), {
                            'JHMs': sample_JHMs,
                            'PAFs': sample_PAFs,

                        })
                    #json save   
                    #print(result_resize)
                    notrack_json = os.path.join(root_dir,'res', "alphaposse_results_resize.json")
                    with open(notrack_json, 'w') as json_file:
                        json_file.write(json.dumps(result_resize))
        '''