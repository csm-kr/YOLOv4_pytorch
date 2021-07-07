import argparse
import torch

# 2. device
device_ids = [0, 1]
device = torch.device('cuda:{}'.format(min(device_ids)) if torch.cuda.is_available() else 'cpu')


def parse(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=273)
    parser.add_argument('--port', type=str, default='2015')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--vis_step', type=int, default=100)
    parser.add_argument('--burn_in', type=int, default=4000)  # 64000 / b_s | b_s == 16 -> 4000 | b_s == 64 -> 1000

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resize', type=int, help='320, 416, 608', default=416)
    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='yolov4_cspdarknet53_coco_exp5')  # FIXME
    parser.add_argument('--conf_thres', type=float, default=0.05)
    parser.add_argument('--start_epoch', type=int, default=0)

    # FIXME choose your dataset root
    # parser.add_argument('--data_root', type=str, default='D:\data\\voc')
    # parser.add_argument('--data_root', type=str, default='D:\data\coco')
    # parser.add_argument('--data_root', type=str, default='/home/cvmlserver5/Sungmin/data/voc')
    parser.add_argument('--data_root', type=str, default='/home/cvmlserver5/Sungmin/data/coco')
    # parser.add_argument('--data_root', type=str, default='/data/voc')

    parser.add_argument('--data_type', type=str, default='coco', help='choose voc or coco')  # FIXME
    parser.add_argument('--num_classes', type=int, default=80)

    opts = parser.parse_args(args)
    print(opts)
    return opts