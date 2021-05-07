import os
import time
import torch
from utils import detect
from evaluator import Evaluator
from config import device


def test(epoch, vis, test_loader, model, criterion, coder, opts):

    # ---------- load ----------
    print('Validation of epoch [{}]'.format(epoch))
    model.eval()
    check_point = torch.load(os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'.format(epoch),
                             map_location=device)
    state_dict = check_point['model_state_dict']
    model.load_state_dict(state_dict)

    tic = time.time()
    sum_loss = 0

    is_coco = hasattr(test_loader.dataset, 'coco')  # if True the set is COCO else VOC
    if is_coco:
        print('COCO dataset evaluation...')
    else:
        print('VOC dataset evaluation...')

    evaluator = Evaluator(data_type=opts.data_type)

    with torch.no_grad():

        for idx, datas in enumerate(test_loader):

            images = datas[0]
            boxes = datas[1]
            labels = datas[2]

            # ---------- cuda ----------
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # ---------- loss ----------
            pred = model(images)
            loss, _ = criterion(pred, boxes, labels)

            sum_loss += loss.item()

            # ---------- eval ----------
            pred_boxes, pred_labels, pred_scores = detect(pred=pred,
                                                          coder=coder,
                                                          opts=opts)

            if opts.data_type == 'voc':
                img_name = datas[3][0]
                img_info = datas[4][0]
                info = (pred_boxes, pred_labels, pred_scores, img_name, img_info)

            elif opts.data_type == 'coco':
                img_id = test_loader.dataset.img_id[idx]
                img_info = test_loader.dataset.coco.loadImgs(ids=img_id)[0]
                coco_ids = test_loader.dataset.coco_ids
                info = (pred_boxes, pred_labels, pred_scores, img_id, img_info, coco_ids)

            evaluator.get_info(info)

            toc = time.time()

            # ---------- print ----------
            if idx % 1000 == 0 or idx == len(test_loader) - 1:
                print('Epoch: [{0}]\t'
                      'Step: [{1}/{2}]\t'
                      'Loss: {loss:.4f}\t'
                      'Time : {time:.4f}\t'
                      .format(epoch,
                              idx, len(test_loader),
                              loss=loss,
                              time=toc - tic))

        mAP = evaluator.evaluate(test_loader.dataset)
        mean_loss = sum_loss / len(test_loader)

        if vis is not None:
            # loss plot
            vis.line(X=torch.ones((1, 2)).cpu() * epoch,  # step
                     Y=torch.Tensor([mean_loss, mAP]).unsqueeze(0).cpu(),
                     win='test_loss',
                     update='append',
                     opts=dict(xlabel='step',
                               ylabel='test',
                               title='test loss',
                               legend=['test Loss', 'mAP']))


if __name__ == '__main__':

    from dataset.coco_dataset import COCO_Dataset
    from dataset.voc_dataset import VOC_Dataset
    from config import device, device_ids
    # from model import YoloV3, Darknet53
    from model import YOLOv4, CSPDarknet53
    from loss import YOLOv4_Loss
    from coder import YOLOv4_Coder
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=265)
    parser.add_argument('--save_path', type=str, default='./saves')
    # parser.add_argument('--save_file_name', type=str, default='yolov3_darknet53_coco')
    parser.add_argument('--save_file_name', type=str, default='yolov4_cspdarknet53_coco')
    parser.add_argument('--conf_thres', type=float, default=0.05)
    # parser.add_argument('--data_root', type=str, default='/home/cvmlserver5/Sungmin/data/coco')
    parser.add_argument('--data_root', type=str, default='D:\data\coco')
    parser.add_argument('--resize', type=int, default=416)
    parser.add_argument('--visualization', type=bool, default=False)
    parser.add_argument('--data_type', type=str, default='coco', help='choose voc or coco')
    parser.add_argument('--num_classes', type=int, default=80)
    test_opts = parser.parse_args()
    print(test_opts)

    if test_opts.data_type == 'voc':
        test_opts.n_classes = 20

    elif test_opts.data_type == 'coco':
        test_opts.n_classes = 80

    # 3. visdom
    vis = None

    train_set = None
    test_set = None

    # 4. data set
    if test_opts.data_type == 'voc':
        test_set = VOC_Dataset(root=test_opts.data_root, split='test', resize=test_opts.resize)
        test_opts.num_classes = 20

    elif test_opts.data_type == 'coco':
        test_set = COCO_Dataset(root=test_opts.data_root, set_name='val2017', split='test', resize=test_opts.resize)
        test_opts.num_classes = 80

    # 5. data loader
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=1,
                                              collate_fn=test_set.collate_fn,
                                              shuffle=False,
                                              num_workers=0,
                                              pin_memory=True)

    # 6. network
    model = YOLOv4(backbone=CSPDarknet53(pretrained=True), num_classes=test_opts.num_classes).to(device)
    # model = YoloV3(baseline=Darknet53(), num_classes=test_opts.num_classes).to(device)
    model = torch.nn.DataParallel(module=model, device_ids=device_ids)

    yolov4_coder = YOLOv4_Coder(test_opts)
    #7. criterion
    criterion = YOLOv4_Loss(coder=yolov4_coder)

    # 12. test
    test(epoch=test_opts.epoch,
         vis=vis,
         test_loader=test_loader,
         model=model,
         criterion=criterion,
         coder=yolov4_coder,
         opts=test_opts)


