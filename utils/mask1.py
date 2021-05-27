import numpy as np
class Mask1():
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
