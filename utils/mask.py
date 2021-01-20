import numpy as np
import cv2


class Mask():
    '''
    def __init__(self, data, gt_mask):
        self.origin_data = data.cpu().numpy()
        self.gt_mask = gt_mask.numpy()
        self.threshold, max_value = self.get_threshold()
        self.mask = self.get_mask()
    '''
    def __init__(self, data):
        self.origin_data = data.cpu().numpy()
        self.threshold, max_value = self.get_threshold()
        self.mask = self.get_mask()

    def get_threshold(self):
        max_value = np.max(self.origin_data)
        threshold = max_value * 0.5
        return threshold, max_value

    def get_mask(self):
        forward_mask = np.ones(self.origin_data.shape)
        backward_mask = np.zeros(self.origin_data.shape)
        mask = np.where(self.origin_data > self.threshold, forward_mask,
                        backward_mask)
        return mask

    def get_box(self):
        mask = self.mask.squeeze() * 255
        contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bbox = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if area > 377 or area < 3000:
                bbox.append([x, y, x + w, y + h])
        return bbox

    def computeIoU(self, gt_box, pr_box):
        xA = max(gt_box[0], pr_box[0])
        yA = max(gt_box[1], pr_box[1])
        xB = min(gt_box[2], pr_box[2])
        yB = min(gt_box[3], pr_box[3])

        interArea = (xB - xA + 1) * (yB - yA + 1)
        boxgtArea = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)
        boxdaArea = (pr_box[2] - pr_box[0] + 1) * (pr_box[3] - pr_box[1] + 1)
        IoU = interArea / float(boxdaArea + boxgtArea - interArea)
        return IoU
