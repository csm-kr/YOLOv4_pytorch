import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch.nn as nn
import torch
from config import device


class Yolov3_Loss(nn.Module):
    def __init__(self, coder):
        super().__init__()

        self.coder = coder
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss(reduction='none')
        self.num_classes = self.coder.num_classes

    def forward(self, pred, gt_boxes, gt_labels):
        """
           :param pred_targets_1: (B, 13, 13, 255)
           :param pred_targets_2: (B, 26, 26, 255)
           :param pred_targets_3: (B, 52, 52, 255)

           :param pred_xy_1 : (B, 13, 13, 2)
           :param pred_wh_1 : (B, 13, 13, 2)

           :param gt_boxes:     (B, 4)
           :param gt_labels:    (B)
           :return:
           """
        batch_size = len(gt_labels)
        pred_targets_l, pred_targets_m, pred_targets_s = pred

        # for Large / Medium/ Small Scale
        size_l, pred_targets_l, pred_txty_l, pred_twth_l, pred_objectness_l, pred_classes_l = self.coder.split_preds(pred_targets_l, for_loss=True)
        size_m, pred_targets_m, pred_txty_m, pred_twth_m, pred_objectness_m, pred_classes_m = self.coder.split_preds(pred_targets_m, for_loss=True)
        size_s, pred_targets_s, pred_txty_s, pred_twth_s, pred_objectness_s, pred_classes_s = self.coder.split_preds(pred_targets_s, for_loss=True)

        # for each scale [gt를 pred의 형식과 맞춘다]
        various_targets = self.coder.build_target(gt_boxes, gt_labels)
        gt_prop_txty_l, gt_twth_l, gt_objectness_l, gt_classes_l, ignore_mask_l = various_targets[0]
        gt_prop_txty_m, gt_twth_m, gt_objectness_m, gt_classes_m, ignore_mask_m = various_targets[1]
        gt_prop_txty_s, gt_twth_s, gt_objectness_s, gt_classes_s, ignore_mask_s = various_targets[2]

        # ----------------------- loss for larage -----------------------
        xy_loss_l = torch.mean((gt_prop_txty_l - pred_txty_l.cpu()) ** 2, dim=-1) * gt_objectness_l.squeeze(-1)
        wh_loss_l = torch.mean((gt_twth_l - pred_twth_l.cpu()) ** 2, dim=-1) * gt_objectness_l.squeeze(-1)
        obj_loss_l = gt_objectness_l * self.bce(pred_objectness_l.cpu(), gt_objectness_l)
        no_obj_loss_l = (1 - gt_objectness_l) * self.bce(pred_objectness_l.cpu(),
                                                         gt_objectness_l) * ignore_mask_l.unsqueeze(-1)
        classes_loss_l = gt_objectness_l * self.bce(pred_classes_l.cpu(), gt_classes_l)

        # ----------------------- loss for medium -----------------------
        xy_loss_m = torch.mean((gt_prop_txty_m - pred_txty_m.cpu()) ** 2, dim=-1) * gt_objectness_m.squeeze(-1)
        wh_loss_m = torch.mean((gt_twth_m - pred_twth_m.cpu()) ** 2, dim=-1) * gt_objectness_m.squeeze(-1)
        obj_loss_m = gt_objectness_m * self.bce(pred_objectness_m.cpu(), gt_objectness_m)
        no_obj_loss_m = (1 - gt_objectness_m) * self.bce(pred_objectness_m.cpu(),
                                                         gt_objectness_m) * ignore_mask_m.unsqueeze(-1)
        classes_loss_m = gt_objectness_m * self.bce(pred_classes_m.cpu(), gt_classes_m)

        # ----------------------- loss for small -----------------------
        xy_loss_s = torch.mean((gt_prop_txty_s - pred_txty_s.cpu()) ** 2, dim=-1) * gt_objectness_s.squeeze(-1)
        wh_loss_s = torch.mean((gt_twth_s - pred_twth_s.cpu()) ** 2, dim=-1) * gt_objectness_s.squeeze(-1)
        obj_loss_s = gt_objectness_s * self.bce(pred_objectness_s.cpu(), gt_objectness_s)
        no_obj_loss_s = (1 - gt_objectness_s) * self.bce(pred_objectness_s.cpu(),
                                                         gt_objectness_s) * ignore_mask_s.unsqueeze(-1)
        classes_loss_s = gt_objectness_s * self.bce(pred_classes_s.cpu(), gt_classes_s)

        # ----------------------- whole losses -----------------------
        xy_loss = 5 * (xy_loss_l.sum() + xy_loss_m.sum() + xy_loss_s.sum()) / batch_size
        wh_loss = 5 * (wh_loss_l.sum() + wh_loss_m.sum() + wh_loss_s.sum()) / batch_size
        obj_loss = 1 * (obj_loss_l.sum() + obj_loss_m.sum() + obj_loss_s.sum()) / batch_size
        no_obj_loss = 0.5 * (no_obj_loss_l.sum() + no_obj_loss_m.sum() + no_obj_loss_s.sum()) / batch_size
        cls_loss = 1 * (classes_loss_l.sum() + classes_loss_m.sum() + classes_loss_s.sum()) / batch_size

        total_loss = xy_loss + wh_loss + obj_loss + no_obj_loss + cls_loss
        return total_loss, (xy_loss, wh_loss, obj_loss, no_obj_loss, cls_loss)

    def forward_(self, pred, gt_boxes, gt_labels):
        """
        :param pred_targets_1: (B, 13, 13, 255)
        :param pred_targets_2: (B, 26, 26, 255)
        :param pred_targets_3: (B, 52, 52, 255)

        :param pred_xy_1 : (B, 13, 13, 2)
        :param pred_wh_1 : (B, 13, 13, 2)

        :param gt_boxes:     (B, 4)
        :param gt_labels:    (B)
        :return:
        """
        batch_size = len(gt_labels)
        pred_targets_1, pred_targets_2, pred_targets_3 = pred

        # Scale 1
        out_size_1 = pred_targets_1.size(1)  # 13, 13
        pred_targets_1 = pred_targets_1.view(-1, out_size_1, out_size_1, 3, 5 + self.num_classes)
        pred_txty_1 = pred_targets_1[..., :2].sigmoid()  # 0, 1 sigmoid(tx, ty) -> bx, by
        pred_twth_1 = pred_targets_1[..., 2:4]
        pred_objectness_1 = pred_targets_1[..., 4].unsqueeze(-1).sigmoid()  # 4        class probability
        pred_classes_1 = pred_targets_1[..., 5:].sigmoid()  # 20 / 80  classes

        # Scale 2
        out_size_2 = pred_targets_2.size(1)
        pred_targets_2 = pred_targets_2.view(-1, out_size_2, out_size_2, 3, 5 + self.num_classes)
        pred_txty_2 = pred_targets_2[..., :2].sigmoid()  # 0, 1 sigmoid(tx, ty) -> bx, by
        pred_twth_2 = pred_targets_2[..., 2:4]
        pred_objectness_2 = pred_targets_2[..., 4].unsqueeze(-1).sigmoid()  # 4        class probability
        pred_classes_2 = pred_targets_2[..., 5:].sigmoid()  # 20 / 80  classes

        # Scale 3
        out_size_3 = pred_targets_3.size(1)
        pred_targets_3 = pred_targets_3.view(-1, out_size_3, out_size_3, 3, 5 + self.num_classes)
        pred_txty_3 = pred_targets_3[..., :2].sigmoid()  # 0, 1 sigmoid(tx, ty) -> bx, by
        pred_twth_3 = pred_targets_3[..., 2:4]
        pred_objectness_3 = pred_targets_3[..., 4].unsqueeze(-1).sigmoid()  # 4        class probability
        pred_classes_3 = pred_targets_3[..., 5:].sigmoid()  # 20 / 80  classes

        # for each scale [gt를 pred의 형식과 맞춘다]
        various_targets = self.coder.build_target(gt_boxes, gt_labels)
        gt_prop_txty_1, gt_twth_1, gt_objectness_1, gt_classes_1, ignore_mask_1 = various_targets[0]
        gt_prop_txty_2, gt_twth_2, gt_objectness_2, gt_classes_2, ignore_mask_2 = various_targets[1]
        gt_prop_txty_3, gt_twth_3, gt_objectness_3, gt_classes_3, ignore_mask_3 = various_targets[2]


        # gt_prop_txty_1, gt_twth_1, gt_objectness_1, gt_classes_1, ignore_mask_1 = self.coder.encode(1, gt_boxes, gt_labels)
        # gt_prop_txty_2, gt_twth_2, gt_objectness_2, gt_classes_2, ignore_mask_2 = self.coder.encode(2, gt_boxes, gt_labels)
        # gt_prop_txty_3, gt_twth_3, gt_objectness_3, gt_classes_3, ignore_mask_3 = self.coder.encode(3, gt_boxes, gt_labels)

        # loss 1
        xy_loss_1 = torch.mean((gt_prop_txty_1 - pred_txty_1.cpu()) ** 2, dim=-1) * gt_objectness_1.squeeze(-1)
        wh_loss_1 = torch.mean((gt_twth_1 - pred_twth_1.cpu()) ** 2, dim=-1) * gt_objectness_1.squeeze(-1)
        obj_loss_1 = gt_objectness_1 * self.bce(pred_objectness_1.cpu(), gt_objectness_1)
        no_obj_loss_1 = (1 - gt_objectness_1) * self.bce(pred_objectness_1.cpu(), gt_objectness_1) * ignore_mask_1.unsqueeze(-1)
        classes_loss_1 = gt_objectness_1 * self.bce(pred_classes_1.cpu(), gt_classes_1)

        # loss 2
        xy_loss_2 = torch.mean((gt_prop_txty_2 - pred_txty_2.cpu()) ** 2, dim=-1) * gt_objectness_2.squeeze(-1)
        wh_loss_2 = torch.mean((gt_twth_2 - pred_twth_2.cpu()) ** 2, dim=-1) * gt_objectness_2.squeeze(-1)
        obj_loss_2 = gt_objectness_2 * self.bce(pred_objectness_2.cpu(), gt_objectness_2)
        no_obj_loss_2 = (1 - gt_objectness_2) * self.bce(pred_objectness_2.cpu(), gt_objectness_2) * ignore_mask_2.unsqueeze(-1)
        classes_loss_2 = gt_objectness_2 * self.bce(pred_classes_2.cpu(), gt_classes_2)

        # loss 3
        xy_loss_3 = torch.mean((gt_prop_txty_3 - pred_txty_3.cpu()) ** 2, dim=-1) * gt_objectness_3.squeeze(-1)
        wh_loss_3 = torch.mean((gt_twth_3 - pred_twth_3.cpu()) ** 2, dim=-1) * gt_objectness_3.squeeze(-1)
        obj_loss_3 = gt_objectness_3 * self.bce(pred_objectness_3.cpu(), gt_objectness_3)
        no_obj_loss_3 = (1 - gt_objectness_3) * self.bce(pred_objectness_3.cpu(), gt_objectness_3) * ignore_mask_3.unsqueeze(-1)
        classes_loss_3 = gt_objectness_3 * self.bce(pred_classes_3.cpu(), gt_classes_3)

        loss1 = 5 * (xy_loss_1.sum() + xy_loss_2.sum() + xy_loss_3.sum()) / batch_size
        loss2 = 5 * (wh_loss_1.sum() + wh_loss_2.sum() + wh_loss_3.sum()) / batch_size
        loss3 = 1 * (obj_loss_1.sum() + obj_loss_2.sum() + obj_loss_3.sum()) / batch_size
        loss4 = 0.5 * (no_obj_loss_1.sum() + no_obj_loss_2.sum() + no_obj_loss_3.sum()) / batch_size
        loss5 = 1 * (classes_loss_1.sum() + classes_loss_2.sum() + classes_loss_3.sum()) / batch_size

        return loss1 + loss2 + loss3 + loss4 + loss5, (loss1, loss2, loss3, loss4, loss5)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='coco')
    loss_opts = parser.parse_args()
    print(loss_opts)

    # model
    image = torch.randn([2, 3, 416, 416]).to(device)

    # coco -> 255 & voc -> 75
    # pred1, pred2, pred3 = YOLOV3_MODEL(image)
    # pred1 : torch.Size([2, 13, 13, 255])
    # pred2 : torch.Size([2, 26, 26, 255])
    # pred3 : torch.Size([2, 52, 52, 255])

    pred1_ = torch.randn([2, 13, 13, 255]).to(device)
    pred2_ = torch.randn([2, 26, 26, 255]).to(device)
    pred3_ = torch.randn([2, 52, 52, 255]).to(device)

    from coder import YOLOv3_Coder

    yolov3_coder = YOLOv3_Coder(loss_opts)

    criterion = Yolov3_Loss(coder=yolov3_coder)

    # ====== 학습 시작 가정 =====
    # torch.Size( [N, 4] )

    # gt = [torch.Tensor([[0.426, 0.158, 0.788, 0.997], [0.0585, 0.1597, 0.8947, 0.8213]]).to(device),
    #       torch.Tensor([[0.002, 0.090, 0.998, 0.867], [0.3094, 0.4396, 0.4260, 0.5440]]).to(device)]
    #
    # # torch.Size( [N] )
    # label = [torch.Tensor([14, 15]).to(device),
    #          torch.Tensor([12, 14]).to(device)]

    gt = [torch.Tensor([[0.4667, 0.5124, 0.7827, 0.7492], [0.4114, 0.4159, 0.8363, 0.6753], [0.6058, 0.4287, 0.6291, 0.4464], [0.6259, 0.4330, 0.6492, 0.4485]]).to(device),
          torch.Tensor([[0.1028, 0.0000, 0.6421, 0.8305], [0.1162, 0.2903, 0.1729, 0.5856]]).to(device)]

    # torch.Size( [N] )
    label = [torch.Tensor([16, 3, 41, 41]).to(device),
             torch.Tensor([39, 61]).to(device)]

    loss = criterion((pred1_, pred2_, pred3_), gt, label)
    loss2 = criterion.forward_((pred1_, pred2_, pred3_), gt, label)

    print("loss : ", loss)
    print("loss2 : ", loss2)