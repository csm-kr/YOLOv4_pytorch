import numpy as np
import torch
from math import sqrt
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from config import device
from utils import cxcy_to_xy


class Anchor(metaclass=ABCMeta):
    def __init__(self, model_name='yolo'):
        self.model_name = model_name.lower()
        assert model_name in ['yolo', 'ssd', 'retina', 'yolov3', 'yolov4']

    @abstractmethod
    def create_anchors(self):
        pass


class YOLOv4_Anchor(Anchor):
    def __init__(self):
        super().__init__()
        self.anchor_whs = {"small": [(10, 13), (16, 30), (33, 23)],
                           "middle": [(30, 61), (62, 45), (59, 119)],
                           "large": [(116, 90), (156, 198), (373, 326)]}

    def anchor_for_scale(self, grid_size, wh):

        center_anchors = []
        for y in range(grid_size):
            for x in range(grid_size):
                cx = x + 0.5
                cy = y + 0.5
                for anchor_wh in wh:
                    w = anchor_wh[0]
                    h = anchor_wh[1]
                    center_anchors.append([cx, cy, w, h])

        print('done!')
        center_anchors_numpy = np.array(center_anchors).astype(np.float32)                          # to numpy  [845, 4]
        center_anchors_tensor = torch.from_numpy(center_anchors_numpy)                              # to tensor [845, 4]
        center_anchors_tensor = center_anchors_tensor.view(grid_size, grid_size, 3, 4).to(device)   # [13, 13, 5, 4]
        return center_anchors_tensor

    def create_anchors(self):

        print('make yolo anchor...')

        wh_large = torch.from_numpy(np.array(self.anchor_whs["large"]) / 32)    # 416 / 32 = 13 - large feature
        wh_middle = torch.from_numpy(np.array(self.anchor_whs["middle"]) / 16)  # 416 / 16 = 26 - medium feature
        wh_small = torch.from_numpy(np.array(self.anchor_whs["small"]) / 8)     # 416 / 8 = 52  - small feature

        center_anchors_large = self.anchor_for_scale(13, wh_large)
        center_anchors_middle = self.anchor_for_scale(26, wh_middle)
        center_anchors_small = self.anchor_for_scale(52, wh_small)

        return center_anchors_large, center_anchors_middle, center_anchors_small


if __name__ == '__main__':
    anchor = YOLOv4_Anchor()

    center_anchors_l, center_anchors_m, center_anchors_s = anchor.create_anchors()

    print("large :", center_anchors_l.shape)
    print("middle :", center_anchors_m.shape)
    print("small :", center_anchors_s.shape)

    anchor_wh = anchor.anchor_whs


    # print(anchor_wh)
    print(anchor_wh["small"])
    # print(anchor_wh["Scale1"][0])


    # a = np.array(anchor_wh["Scale1"]) / 32
    # b = np.array(anchor_wh["Scale2"]) / 16
    # c = np.array(anchor_wh["Scale3"]) / 8
    # print(a)
    # wh = torch.from_numpy(a)
    # print(wh.shape)

