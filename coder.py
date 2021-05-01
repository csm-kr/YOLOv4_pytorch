import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import torch
from math import sqrt
from abc import ABCMeta, abstractmethod
from anchor import YOLOv3_Anchor
import torch.nn.functional as F
from config import device
from utils import cxcy_to_xy, xy_to_cxcy, find_jaccard_overlap


class Coder(metaclass=ABCMeta):

    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def decode(self):
        pass


class YOLOv3_Coder(Coder):
    def __init__(self, opts):
        super().__init__()
        self.data_type = opts.data_type
        self.anchor = YOLOv3_Anchor()

        self.anchor_whs = self.anchor.anchor_whs
        self.center_anchor_l, self.center_anchor_m, self.center_anchor_s = self.anchor.create_anchors()  # Anchor

        assert self.data_type in ['voc', 'coco']
        if self.data_type == 'voc':
            self.num_classes = 20
        elif self.data_type == 'coco':
            self.num_classes = 80

    def assign_anchors_to_device(self):
        self.center_anchor_l = self.center_anchor_l.to(device)
        self.center_anchor_m = self.center_anchor_m.to(device)
        self.center_anchor_s = self.center_anchor_s.to(device)

    def assign_anchors_to_cpu(self):
        self.center_anchor_l = self.center_anchor_l.to('cpu')
        self.center_anchor_m = self.center_anchor_m.to('cpu')
        self.center_anchor_s = self.center_anchor_s.to('cpu')

    def build_target(self, gt_boxes, gt_labels):

        batch_size = len(gt_labels)

        # ---------------- 1. container 만들기 ----------------
        # large target container
        ignore_mask_l = torch.zeros([batch_size, 13, 13, 3])
        gt_prop_txty_l = torch.zeros([batch_size, 13, 13, 3, 2])    # a proportion between (0 ~ 1) in a cell
        gt_twth_l = torch.zeros([batch_size, 13, 13, 3, 2])     # ratio of gt box and anchor box
        gt_objectness_l = torch.zeros([batch_size, 13, 13, 3, 1])   # maximum iou anchor (a obj assign a anc)
        gt_classes_l = torch.zeros([batch_size, 13, 13, 3, self.num_classes])   # one-hot encoded class label

        # middle target container
        ignore_mask_m = torch.zeros([batch_size, 26, 26, 3])
        gt_prop_txty_m = torch.zeros([batch_size, 26, 26, 3, 2])    # a proportion between (0 ~ 1) in a cell
        gt_twth_m = torch.zeros([batch_size, 26, 26, 3, 2])     # ratio of gt box and anchor box
        gt_objectness_m = torch.zeros([batch_size, 26, 26, 3, 1])   # maximum iou anchor (a obj assign a anc)
        gt_classes_m = torch.zeros([batch_size, 26, 26, 3, self.num_classes])   # one-hot encoded class label

        # small target container
        ignore_mask_s = torch.zeros([batch_size, 52, 52, 3])
        gt_prop_txty_s = torch.zeros([batch_size, 52, 52, 3, 2])    # a proportion between (0 ~ 1) in a cell
        gt_twth_s = torch.zeros([batch_size, 52, 52, 3, 2])     # ratio of gt box and anchor box
        gt_objectness_s = torch.zeros([batch_size, 52, 52, 3, 1])   # maximum iou anchor (a obj assign a anc)
        gt_classes_s = torch.zeros([batch_size, 52, 52, 3, self.num_classes])   # one-hot encoded class label

        # ---------------- 2. anchor 만들기 ----------------
        center_anchor_l = self.center_anchor_l
        corner_anchor_l = cxcy_to_xy(center_anchor_l).view(13 * 13 * 3, 4)

        center_anchor_m = self.center_anchor_m
        corner_anchor_m = cxcy_to_xy(center_anchor_m).view(26 * 26 * 3, 4)

        center_anchor_s = self.center_anchor_s
        corner_anchor_s = cxcy_to_xy(center_anchor_s).view(52 * 52 * 3, 4)

        for b in range(batch_size):

            label = gt_labels[b]
            corner_gt_box = gt_boxes[b]                             # (0 ~ 1) x1 y1 x2 y2
            size = [13, 26, 52]

            center_gt_box = xy_to_cxcy(corner_gt_box)               # (0 ~ 1) cx cy w h

            # must consider 3 scales --------------------------------------------------------------------------------
            # scaled corner gt box for iou
            scaled_corner_gt_box_l = corner_gt_box * float(size[0])  # (0 ~ 13 - 1/32) x1 y1 x2 y2
            scaled_corner_gt_box_m = corner_gt_box * float(size[1])  # (0 ~ 26 - 1/16) x1 y1 x2 y2
            scaled_corner_gt_box_s = corner_gt_box * float(size[2])  # (0 ~ 52 - 1/ 8) x1 y1 x2 y2

            iou_anchors_gt_l = find_jaccard_overlap(corner_anchor_l, scaled_corner_gt_box_l)  # [13 * 13 * 3, # obj]
            iou_anchors_gt_m = find_jaccard_overlap(corner_anchor_m, scaled_corner_gt_box_m)  # [26 * 26 * 3, # obj]
            iou_anchors_gt_s = find_jaccard_overlap(corner_anchor_s, scaled_corner_gt_box_s)  # [52 * 52 * 3, # obj]

            # scaled center gt box for assign
            scaled_center_gt_box_l = center_gt_box * float(size[0])  # (0 ~ out_size) x1 y1 x2 y2
            scaled_center_gt_box_m = center_gt_box * float(size[1])  # (0 ~ out_size) x1 y1 x2 y2
            scaled_center_gt_box_s = center_gt_box * float(size[2])  # (0 ~ out_size) x1 y1 x2 y2

            # large
            bxby_l = scaled_center_gt_box_l[..., :2]        # [obj, 2] - cxcy
            proportion_of_xy_l = bxby_l - bxby_l.floor()    # [obj, 2] - 0 ~ 1
            bwbh_l = scaled_center_gt_box_l[..., 2:]         # [obj, 2] - wh

            # medium
            bxby_m = scaled_center_gt_box_m[..., :2]        # [obj, 2] - cxcy
            proportion_of_xy_m = bxby_m - bxby_m.floor()    # [obj, 2] - 0 ~ 1
            bwbh_m = scaled_center_gt_box_m[..., 2:]        # [obj, 2] - wh

            # small
            bxby_s = scaled_center_gt_box_s[..., :2]        # [obj, 2] - cxcy
            proportion_of_xy_s = bxby_s - bxby_s.floor()    # [obj, 2] - 0 ~ 1
            bwbh_s = scaled_center_gt_box_s[..., 2:]        # [obj, 2] - wh

            iou_anchors_gt_l = iou_anchors_gt_l.view(size[0], size[0], 3, -1)
            iou_anchors_gt_m = iou_anchors_gt_m.view(size[1], size[1], 3, -1)
            iou_anchors_gt_s = iou_anchors_gt_s.view(size[2], size[2], 3, -1)

            num_obj = corner_gt_box.size(0)

            for n_obj in range(num_obj):
                # iterate each obj, find which anchor is best anchor

                # iou_anchors_gt_l[..., n_obj]  # [13 ,13 ,3, # obj] --> [13, 13, 3]
                # iou_anchors_gt_m[..., n_obj]  # [26 ,26 ,3, # obj] --> [26, 26, 3]
                # iou_anchors_gt_s[..., n_obj]  # [52 ,52 ,3, # obj] --> [52, 52, 3]

                # find best anchor
                best_idx = torch.FloatTensor(
                    [iou_anchors_gt_l[..., n_obj].max(),
                     iou_anchors_gt_m[..., n_obj].max(),
                     iou_anchors_gt_s[..., n_obj].max()]).argmax()

                if best_idx == 0:
                    cx, cy = bxby_l[n_obj]
                    cx = int(cx)
                    cy = int(cy)

                    max_iou, max_idx = iou_anchors_gt_l[cy, cx, :, n_obj].max(0)  # which anchor has maximum iou?
                    # print("max_iou : ", max_iou)
                    j = max_idx  # j is idx.
                    # # j-th anchor
                    gt_objectness_l[b, cy, cx, j, 0] = 1
                    gt_prop_txty_l[b, cy, cx, j, :] = proportion_of_xy_l[n_obj]

                    ratio_of_wh_l = bwbh_l[n_obj] / torch.from_numpy(np.array(self.anchor_whs["large"][j]) / 32). \
                        to(device)

                    gt_twth_l[b, cy, cx, j, :] = torch.log(ratio_of_wh_l)
                    gt_classes_l[b, cy, cx, j, int(label[n_obj].item())] = 1

                elif best_idx == 1:
                    cx, cy = bxby_m[n_obj]
                    cx = int(cx)
                    cy = int(cy)

                    max_iou, max_idx = iou_anchors_gt_m[cy, cx, :, n_obj].max(0)  # which anchor has maximum iou?
                    # print("max_iou : ", max_iou)
                    j = max_idx  # j is idx.
                    # # j-th anchor
                    gt_objectness_m[b, cy, cx, j, 0] = 1
                    gt_prop_txty_m[b, cy, cx, j, :] = proportion_of_xy_m[n_obj]

                    ratio_of_wh_m = bwbh_m[n_obj] / torch.from_numpy(np.array(self.anchor_whs["middle"][j]) / 16). \
                        to(device)

                    gt_twth_m[b, cy, cx, j, :] = torch.log(ratio_of_wh_m)
                    gt_classes_m[b, cy, cx, j, int(label[n_obj].item())] = 1

                elif best_idx == 2:
                    cx, cy = bxby_s[n_obj]
                    cx = int(cx)
                    cy = int(cy)

                    max_iou, max_idx = iou_anchors_gt_s[cy, cx, :, n_obj].max(0)  # which anchor has maximum iou?
                    # print("max_iou : ", max_iou)
                    j = max_idx  # j is idx.
                    # # j-th anchor
                    gt_objectness_s[b, cy, cx, j, 0] = 1
                    gt_prop_txty_s[b, cy, cx, j, :] = proportion_of_xy_s[n_obj]

                    ratio_of_wh_s = bwbh_s[n_obj] / torch.from_numpy(np.array(self.anchor_whs["small"][j]) / 8). \
                        to(device)

                    gt_twth_s[b, cy, cx, j, :] = torch.log(ratio_of_wh_s)
                    gt_classes_s[b, cy, cx, j, int(label[n_obj].item())] = 1

                # ignore_mask
                ignore_mask_l[b] = (iou_anchors_gt_l.max(-1)[0] < 0.5)
                ignore_mask_m[b] = (iou_anchors_gt_m.max(-1)[0] < 0.5)
                ignore_mask_s[b] = (iou_anchors_gt_s.max(-1)[0] < 0.5)

        targes = [[gt_prop_txty_l, gt_twth_l, gt_objectness_l, gt_classes_l, ignore_mask_l],
                  [gt_prop_txty_m, gt_twth_m, gt_objectness_m, gt_classes_m, ignore_mask_m],
                  [gt_prop_txty_s, gt_twth_s, gt_objectness_s, gt_classes_s, ignore_mask_s]]

        return targes

    def encode(self, gt_boxes, gt_labels):
        return

    def decode(self, gcxgcys, direct_pred=True):
        """
        gcxgcys : ([B, 13, 13, 5, 4],[B, 26, 26, 5, 4],[B, 52, 52, 5, 4])
        """
        gcxgcy_l, gcxgcy_m, gcxgcy_s = gcxgcys
        cxcy_l = torch.sigmoid(gcxgcy_l[..., :2]) + self.center_anchor_l[..., :2].floor()
        wh_l = torch.exp(gcxgcy_l[..., 2:]) * self.center_anchor_l[..., 2:]
        cxcywh_l = torch.cat([cxcy_l, wh_l], dim=-1)

        cxcy_m = torch.sigmoid(gcxgcy_m[..., :2]) + self.center_anchor_m[..., :2].floor()
        wh_m = torch.exp(gcxgcy_m[..., 2:]) * self.center_anchor_m[..., 2:]
        cxcywh_m = torch.cat([cxcy_m, wh_m], dim=-1)

        cxcy_s = torch.sigmoid(gcxgcy_s[..., :2]) + self.center_anchor_s[..., :2].floor()
        wh_s = torch.exp(gcxgcy_s[..., 2:]) * self.center_anchor_s[..., 2:]
        cxcywh_s = torch.cat([cxcy_s, wh_s], dim=-1)

        return cxcywh_l, cxcywh_m, cxcywh_s

    # def decode_(self, pred):
    #
    #     pred_targets_1, pred_targets_2, pred_targets_3 = pred
    #
    #     # Scale 1
    #     out_size_1 = pred_targets_1.size(1)
    #     pred_targets_1 = pred_targets_1.view(-1, out_size_1, out_size_1, 3, 5 + self.num_classes)
    #     pred_txty_1 = pred_targets_1[..., :2].sigmoid()                     # 0, 1     sigmoid(tx, ty) -> bx, by
    #     pred_twth_1 = pred_targets_1[..., 2:4]                              # 2, 3
    #     pred_objectness_1 = pred_targets_1[..., 4].unsqueeze(-1).sigmoid()  # 4        class probability
    #     pred_classes_1 = pred_targets_1[..., 5:].sigmoid()                  # 20 / 80  classes
    #
    #     # Scale 2
    #     out_size_2 = pred_targets_2.size(1)
    #     pred_targets_2 = pred_targets_2.view(-1, out_size_2, out_size_2, 3, 5 + self.num_classes)
    #     pred_txty_2 = pred_targets_2[..., :2].sigmoid()                     # 0, 1 sigmoid(tx, ty) -> bx, by
    #     pred_twth_2 = pred_targets_2[..., 2:4]                              # 2, 3
    #     pred_objectness_2 = pred_targets_2[..., 4].unsqueeze(-1).sigmoid()  # 4        class probability
    #     pred_classes_2 = pred_targets_2[..., 5:].sigmoid()                  # 20 / 80  classes
    #
    #     # Scale 3
    #     out_size_3 = pred_targets_3.size(1)
    #     pred_targets_3 = pred_targets_3.view(-1, out_size_3, out_size_3, 3, 5 + self.num_classes)
    #     pred_txty_3 = pred_targets_3[..., :2].sigmoid()                     # 0, 1 sigmoid(tx, ty) -> bx, by
    #     pred_twth_3 = pred_targets_3[..., 2:4]                              # 2, 3
    #     pred_objectness_3 = pred_targets_3[..., 4].unsqueeze(-1).sigmoid()  # 4        class probability
    #     pred_classes_3 = pred_targets_3[..., 5:].sigmoid()                  # 20 / 80  classes
    #
    #     self.center_anchor_1 = self.center_anchor_l.to(device)           # 13 , 13
    #     self.center_anchor_2 = self.center_anchor_m.to(device)           # 26 , 26
    #     self.center_anchor_3 = self.center_anchor_s.to(device)           # 52 , 52
    #
    #     center_anchors_xy_1 = self.center_anchor_1[..., :2]  # torch.Size([13, 13, 3, 2])
    #     center_anchors_wh_1 = self.center_anchor_1[..., 2:]  # torch.Size([13, 13, 3, 2])
    #
    #     center_anchors_xy_2 = self.center_anchor_2[..., :2]  # torch.Size([26, 26, 3, 2])
    #     center_anchors_wh_2 = self.center_anchor_2[..., 2:]  # torch.Size([26, 26, 3, 2])
    #
    #     center_anchors_xy_3 = self.center_anchor_3[..., :2]  # torch.Size([52, 52, 3, 2])
    #     center_anchors_wh_3 = self.center_anchor_3[..., 2:]  # torch.Size([52, 52, 3, 2])
    #
    #     pred_bbox_xy_1 = center_anchors_xy_1.floor().expand_as(pred_txty_1) + pred_txty_1
    #     pred_bbox_wh_1 = center_anchors_wh_1.expand_as(pred_twth_1) * pred_twth_1.exp()
    #     pred_bbox_1 = torch.cat([pred_bbox_xy_1, pred_bbox_wh_1], dim=-1)                    # [B, 13, 13, 3, 4]
    #     pred_bbox_1 = pred_bbox_1.view(-1, out_size_1 * out_size_1 * 3, 4) / out_size_1      # [B, 507, 4]
    #     pred_cls_1 = pred_classes_1.view(-1, out_size_1 * out_size_1 * 3, self.num_classes)  # [B, 507, 80]
    #     pred_conf_1 = pred_objectness_1.view(-1, out_size_1 * out_size_1 * 3)                                # [B, 507]
    #
    #     pred_bbox_xy_2 = center_anchors_xy_2.floor().expand_as(pred_txty_2) + pred_txty_2
    #     pred_bbox_wh_2 = center_anchors_wh_2.expand_as(pred_twth_2) * pred_twth_2.exp()
    #     pred_bbox_2 = torch.cat([pred_bbox_xy_2, pred_bbox_wh_2], dim=-1)                    # [B, 26, 26, 3, 4]
    #     pred_bbox_2 = pred_bbox_2.view(-1, out_size_2 * out_size_2 * 3, 4) / out_size_2      # [B, 2028, 4]
    #     pred_cls_2 = pred_classes_2.view(-1, out_size_2 * out_size_2 * 3, self.num_classes)  # [B, 2028, 80]
    #     pred_conf_2 = pred_objectness_2.view(-1, out_size_2 * out_size_2 * 3)                # [B, 2028]
    #
    #     pred_bbox_xy_3 = center_anchors_xy_3.floor().expand_as(pred_txty_3) + pred_txty_3
    #     pred_bbox_wh_3 = center_anchors_wh_3.expand_as(pred_twth_3) * pred_twth_3.exp()
    #     pred_bbox_3 = torch.cat([pred_bbox_xy_3, pred_bbox_wh_3], dim=-1)                    # [B, 52, 52, 3, 4]
    #     pred_bbox_3 = pred_bbox_3.view(-1, out_size_3 * out_size_3 * 3, 4) / out_size_3      # [B, 8112, 4]
    #     pred_cls_3 = pred_classes_3.view(-1, out_size_3 * out_size_3 * 3, self.num_classes)  # [B, 8112, 80]
    #     pred_conf_3 = pred_objectness_3.view(-1, out_size_3 * out_size_3 * 3)                # [B, 8112]
    #
    #     pred_bbox = torch.cat([pred_bbox_1, pred_bbox_2, pred_bbox_3], dim=1)                # [B, 10647, 4]
    #     pred_cls = torch.cat([pred_cls_1, pred_cls_2, pred_cls_3], dim=1)                    # [B, 10647, 80]
    #     pred_conf = torch.cat([pred_conf_1, pred_conf_2, pred_conf_3], dim=1)                # [B, 10647]
    #
    #     return pred_bbox, pred_cls, pred_conf
    #
    # def post_processing(self, pred, is_demo=False):
    #     if is_demo:
    #         self.assign_anchors_to_cpu()
    #         # pred = [pred[0].to('cpu'), pred[1].to('cpu'), pred[2].to('cpu')]
    #
    #     pred_bbox, pred_cls, pred_conf = self.decode_(pred)
    #     # [1, 10647, 4]
    #     # [1, 10647, 80]
    #     # [1, 10647]
    #
    #     # output bbox of yolo model is center coord
    #     pred_bboxes = cxcy_to_xy(pred_bbox).squeeze()
    #     pred_scores = (pred_cls * pred_conf.unsqueeze(-1)).squeeze()
    #     return pred_bboxes, pred_scores

    def postprocessing(self, pred, is_demo=False):

        pred_targets_l, pred_targets_m, pred_targets_s = pred

        if is_demo:
            self.assign_anchors_to_cpu()
            pred_targets_l = pred_targets_l.to('cpu')
            pred_targets_m = pred_targets_m.to('cpu')
            pred_targets_s = pred_targets_s.to('cpu')

        # for Large / Medium/ Small Scale
        size_l, pred_targets_l, pred_bbox_l, pred_objectness_l, pred_classes_l = self.split_preds(pred_targets_l)
        size_m, pred_targets_m, pred_bbox_m, pred_objectness_m, pred_classes_m = self.split_preds(pred_targets_m)
        size_s, pred_targets_s, pred_bbox_s, pred_objectness_s, pred_classes_s = self.split_preds(pred_targets_s)

        # box decode
        pred_bbox_l, pred_bbox_m, pred_bbox_s = self.decode((pred_bbox_l, pred_bbox_m, pred_bbox_s))

        # Clean up the code shape
        pred_bbox_l = pred_bbox_l.view(-1, size_l * size_l * 3, 4) / size_l                  # [B, 507, 4]
        pred_cls_l = pred_classes_l.view(-1, size_l * size_l * 3, self.num_classes)          # [B, 507, 80]
        pred_conf_l = pred_objectness_l.view(-1, size_l * size_l * 3)                        # [B, 507]

        pred_bbox_m = pred_bbox_m.view(-1, size_m * size_m * 3, 4) / size_m                  # [B, 2028, 4]
        pred_cls_m = pred_classes_m.view(-1, size_m * size_m * 3, self.num_classes)          # [B, 2028, 80]
        pred_conf_m = pred_objectness_m.view(-1, size_m * size_m * 3)                        # [B, 2028]

        pred_bbox_s = pred_bbox_s.view(-1, size_s * size_s * 3, 4) / size_s                  # [B, 8112, 4]
        pred_cls_s = pred_classes_s.view(-1, size_s * size_s * 3, self.num_classes)          # [B, 8112, 80]
        pred_conf_s = pred_objectness_s.view(-1, size_s * size_s * 3)                        # [B, 8112]

        # concat predictions
        pred_bbox = torch.cat([pred_bbox_l, pred_bbox_m, pred_bbox_s], dim=1)                # [B, 10647, 4]
        pred_cls = torch.cat([pred_cls_l, pred_cls_m, pred_cls_s], dim=1)                    # [B, 10647, 80]
        pred_conf = torch.cat([pred_conf_l, pred_conf_m, pred_conf_s], dim=1)                # [B, 10647]

        # make pred boxes and pred scores using conditional prob concepts
        pred_bboxes = cxcy_to_xy(pred_bbox).squeeze()
        pred_scores = (pred_cls * pred_conf.unsqueeze(-1)).squeeze()

        return pred_bboxes, pred_scores

    def split_preds(self, pred, for_loss=False):

        if for_loss:
            out_size = pred.size(1)
            pred = pred.view(-1, out_size, out_size, 3, 5 + self.num_classes)
            pred_txty = pred[..., :2].sigmoid()  # 0, 1 sigmoid(tx, ty) -> bx, by
            pred_twth = pred[..., 2:4]
            pred_objectness = pred[..., 4].unsqueeze(-1).sigmoid()  # 4            class probability
            pred_classes = pred[..., 5:].sigmoid()  # 20 / 80      classes

            return out_size, pred, pred_txty, pred_twth, pred_objectness, pred_classes

        out_size = pred.size(1)
        pred = pred.view(-1, out_size, out_size, 3, 5 + self.num_classes)
        pred_bbox = pred[..., :4]                                            # 0, 1, 2, 3   xywh
        pred_objectness = pred[..., 4].unsqueeze(-1).sigmoid()               # 4            class probability
        pred_classes = pred[..., 5:].sigmoid()                               # 20 / 80      classes

        return out_size, pred, pred_bbox, pred_objectness, pred_classes


if __name__ == '__main__':
    coder = YOLOv3_Coder('coco')
    coder.assign_anchors_to_device()