from os.path import join
import math
import json
import os
import cv2
import scipy.io as sio
import numpy as np
np.set_printoptions(threshold=np.inf)


def get_heatmap(ori_size, pose, target_size):
    width, height = ori_size
    heatmap = np.zeros((pose.shape[1] + 1, height, width), dtype=np.float32)

    # print(pose[0][0])
    for person in range(pose.shape[0]):
        for idx, point in enumerate(pose[person]):
            if point[0] <= 0 or point[1] <= 0:
                continue
            point = point[0:2]
            heatmap = put_heatmap(heatmap, idx, point[0:2], 1.0)
    # print(heatmap[24, 492:540, 779:828])
    # sio.savemat("D:\Project\Person-in-WiFi\dataset\mask_resize\\" + "heatmap.mat", {'heatmap': np.array(heatmap)})
    heatmap = heatmap.transpose((1, 2, 0))
    heatmap[:, :, -1] = np.clip(1 - np.amax(heatmap, axis=2), 0.0, 1.0)
    # print(heatmap)
    if target_size:
        heatmap = cv2.resize(heatmap, target_size)
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


def get_vectormap(ori_size, pose, target_size):
    width, height = ori_size
    # print(pose.shape[1])
    limb = list(
        zip([
            0, 1, 0, 2, 5, 5, 7, 6, 8, 11, 12, 13, 14
        ], [
            1, 3, 2, 4, 6, 7, 9, 8, 10, 13, 14, 15, 16
        ]))
    vectormap = np.zeros(((len(limb)) * 2, height, width),dtype=np.float32)
    countmap = np.zeros((len(limb), height, width), dtype=np.int16)
    for joins in range(pose.shape[0]):
        for plane_idx, (j_idx1, j_idx2) in enumerate(limb):
            # j_idx1 -= 1
            # j_idx2 -= 1

            center_from = pose[joins][j_idx1]
            center_to = pose[joins][j_idx2]
            if center_from[0] < -100 or center_from[1] < -100 or center_to[
                    0] < -100 or center_to[1] < -100 or center_from[
                        2] <= 0 or center_to[2] <= 0:
                continue
            vectormap = put_vectormap(vectormap, countmap, plane_idx,
                                      center_from, center_to)

    vectormap = vectormap.transpose(1, 2, 0)
    #print(vectormap.shape)
    nonzeros = np.nonzero(countmap)
    # Image.fromarray(np.sum(countmap, axis=0).astype(np.float) * 255).show()
    # print(nonzeros)
    for p, y, x in zip(nonzeros[0], nonzeros[1], nonzeros[2]):
        if countmap[p][y][x] <= 0:
            continue
        vectormap[y][x][p * 2 + 0] /= countmap[p][y][x]
        vectormap[y][x][p * 2 + 1] /= countmap[p][y][x]

    if target_size:
        vectormap = cv2.resize(vectormap, target_size)
    return vectormap.astype(np.float16)


def put_vectormap(vectormap,
                  countmap,
                  plane_idx,
                  center_from,
                  center_to,
                  threshold=2):
    _, height, width = vectormap.shape[:3]
    # print(height,width)
    vec_x = center_to[0] - center_from[0]
    vec_y = center_to[1] - center_from[1]
    min_x = max(0, int(min(center_from[0], center_to[0]) - threshold))
    min_y = max(0, int(min(center_from[1], center_to[1]) - threshold))

    max_x = min(width, int(max(center_from[0], center_to[0]) + threshold))
    max_y = min(height, int(max(center_from[1], center_to[1]) + threshold))
    norm = math.sqrt(vec_x**2 + vec_y**2)
    if norm == 0:
        return vectormap

    vec_x /= norm
    vec_y /= norm

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            bec_x = x - center_from[0]
            bec_y = y - center_from[1]
            dist = abs(bec_x * vec_y - bec_y * vec_x)

            if dist > threshold:
                continue

            countmap[plane_idx][y][x] += 1

            vectormap[plane_idx * 2 + 0][y][x] = vec_x
            vectormap[plane_idx * 2 + 1][y][x] = vec_y

    return vectormap


if __name__ == "__main__":
    root_dir = "/home/public/b509/code/g19/jyf/mypro/PersonWiFi3/dataset"
    pose_dir = join(root_dir, "res")
    print(pose_dir)

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