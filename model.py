import torch
import wget
import os
import torch.nn as nn
from utils import bar_custom
from config import device
import numpy as np
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, hidden_channel=None):
        if hidden_channel is None:
            hidden_channel = in_channel
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channel, hidden_channel, 1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(hidden_channel),
                                      Mish(),
                                      nn.Conv2d(hidden_channel, in_channel, 3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(in_channel),
                                      Mish(),
                                      )

    def forward(self, x):
        residual = x
        x = self.features(x)
        x += residual
        return x


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class CSPBlock(nn.Module):
    def __init__(self, in_channel, is_first=False, num_blocks=1):
        super().__init__()
        self.part1_conv = nn.Sequential(nn.Conv2d(in_channel, in_channel//2, 1, stride=1, padding=0, bias=False),
                                        nn.BatchNorm2d(in_channel//2),
                                        Mish())
        self.part2_conv = nn.Sequential(nn.Conv2d(in_channel, in_channel//2, 1, stride=1, padding=0, bias=False),
                                        nn.BatchNorm2d(in_channel//2),
                                        Mish())
        self.features = nn.Sequential(*[ResidualBlock(in_channel=in_channel//2) for _ in range(num_blocks)])
        self.transition1_conv = nn.Sequential(nn.Conv2d(in_channel//2, in_channel//2, 1, stride=1, padding=0, bias=False),
                                              nn.BatchNorm2d(in_channel//2),
                                              Mish())
        self.transition2_conv = nn.Sequential(nn.Conv2d(in_channel, in_channel, 1, stride=1, padding=0, bias=False),
                                              nn.BatchNorm2d(in_channel),
                                              Mish())
        if is_first:
            self.part1_conv = nn.Sequential(nn.Conv2d(in_channel, in_channel, 1, stride=1, padding=0, bias=False),
                                            nn.BatchNorm2d(in_channel),
                                            Mish())
            self.part2_conv = nn.Sequential(nn.Conv2d(in_channel, in_channel, 1, stride=1, padding=0, bias=False),
                                            nn.BatchNorm2d(in_channel),
                                            Mish())
            self.features = nn.Sequential(*[ResidualBlock(in_channel=in_channel,
                                                          hidden_channel=in_channel//2) for _ in range(num_blocks)])
            self.transition1_conv = nn.Sequential(nn.Conv2d(in_channel, in_channel, 1, stride=1, padding=0, bias=False),
                                                  nn.BatchNorm2d(in_channel),
                                                  Mish())
            self.transition2_conv = nn.Sequential(nn.Conv2d(2 * in_channel, in_channel, 1, stride=1, padding=0, bias=False),
                                                  nn.BatchNorm2d(in_channel),
                                                  Mish())

    def forward(self, x):
        # split features
        part1 = self.part1_conv(x)
        part2 = self.part2_conv(x)

        # residual
        residual = part2
        part2 = self.features(part2)
        part2 += residual
        part2 = self.transition1_conv(part2)

        x = self.transition2_conv(torch.cat([part1, part2], dim=1))
        return x


class CSPDarknet53(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super().__init__()

        self.num_classes = num_classes
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Mish(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            Mish(),
            CSPBlock(64, is_first=True, num_blocks=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            Mish(),
            CSPBlock(128, num_blocks=2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            Mish(),
            CSPBlock(256, num_blocks=8),
        )

        self.features2 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            Mish(),
            CSPBlock(512, num_blocks=8),
        )

        self.features3 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            Mish(),
            CSPBlock(1024, num_blocks=4),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)
        self.init_layer()
        if pretrained:
            self.yolov4_weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
            self.pretrained_path = "./pretrained"
            os.makedirs(self.pretrained_path, exist_ok=True)
            self.pretrained_file = os.path.join(self.pretrained_path, 'yolov4.weights')
            if os.path.exists(self.pretrained_file):
                print("weight already exist!")
            else:
                wget.download(url=self.yolov4_weights_url, out=self.pretrained_path, bar=bar_custom)
                print('')
                print("Done!")
            self.load_darknet_weights(self.pretrained_file)

        print("num_params : ", self.count_parameters())

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def init_layer(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.gap(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        ptr = 0
        conv_layer = None
        # refer to https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/models.py
        for i, module in enumerate(self.modules()):
            if isinstance(module, nn.Conv2d):
                conv_layer = module
            if isinstance(module, nn.BatchNorm2d):
                bn_layer = module
                num_b = bn_layer.bias.numel()  # Number of biases

                # Bias
                bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b

                # Weight
                bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b

                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b

                # Running Var
                bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                continue
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w


# https://github.com/ultralytics/yolov5/blob/850970e081687df6427898948a27df37ab4de5d3/models/common.py#L139
class SPPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(1024, 512, 1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(512),
                        Mish(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2048, 2048, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2048),
            Mish(),
        )

        self.maxpool5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5//2)
        self.maxpool9 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9//2)
        self.maxpool13 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13//2)

    def forward(self, x):
        x = self.conv1(x)   # torch.Size([1, 512, 16, 16])
        maxpool5 = self.maxpool5(x)
        maxpool9 = self.maxpool9(x)
        maxpool13 = self.maxpool13(x)
        x = torch.cat([x, maxpool5, maxpool9, maxpool13], dim=1)
        x = self.conv2(x)
        return x


class PANet(nn.Module):
    def __init__(self):
        super(PANet, self).__init__()

        self.p52d5 = nn.Sequential(nn.Conv2d(2048, 512, 1, stride=1, padding=0, bias=False),
                                   nn.BatchNorm2d(512),
                                   Mish(),
                                   nn.Conv2d(512, 1024, 3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(1024),
                                   Mish(),
                                   nn.Conv2d(1024, 512, 1, stride=1, padding=0, bias=False),
                                   nn.BatchNorm2d(512),
                                   Mish(),
                                   )

        self.p42p4_ = nn.Sequential(nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False),
                                    nn.BatchNorm2d(256),
                                    Mish(),
                                    )

        self.p32p3_ = nn.Sequential(nn.Conv2d(256, 128, 1, stride=1, padding=0, bias=False),
                                    nn.BatchNorm2d(128),
                                    Mish(),
                                    )

        self.d5_p4_2d4 = nn.Sequential(nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(256),
                                       Mish(),
                                       nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(512),
                                       Mish(),
                                       nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(256),
                                       Mish(),
                                       nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(512),
                                       Mish(),
                                       nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(256),
                                       Mish(),
                                       )

        self.d4_p3_2d3 = nn.Sequential(nn.Conv2d(256, 128, 1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(128),
                                       Mish(),
                                       nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       Mish(),
                                       nn.Conv2d(256, 128, 1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(128),
                                       Mish(),
                                       nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       Mish(),
                                       nn.Conv2d(256, 128, 1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(128),
                                       Mish(),
                                       )

        self.d52d5_ = nn.Sequential(nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False),
                                    nn.BatchNorm2d(256),
                                    Mish(),
                                    nn.Upsample(scale_factor=2)
                                    )

        self.d42d4_ = nn.Sequential(nn.Conv2d(256, 128, 1, stride=1, padding=0, bias=False),
                                    nn.BatchNorm2d(128),
                                    Mish(),
                                    nn.Upsample(scale_factor=2)
                                    )

        self.u32u3_ = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(256),
                                    Mish())

        self.u42u4_ = nn.Sequential(nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(512),
                                    Mish())

        self.d4u3_2u4 = nn.Sequential(nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(256),
                                      Mish(),

                                      nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(512),
                                      Mish(),

                                      nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(256),
                                      Mish(),

                                      nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(512),
                                      Mish(),

                                      nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(256),
                                      Mish(),
                                      )

        self.d5u4_2u5 = nn.Sequential(nn.Conv2d(1024, 512, 1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(512),
                                      Mish(),

                                      nn.Conv2d(512, 1024, 3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(1024),
                                      Mish(),

                                      nn.Conv2d(1024, 512, 1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(512),
                                      Mish(),

                                      nn.Conv2d(512, 1024, 3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(1024),
                                      Mish(),

                                      nn.Conv2d(1024, 512, 1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(512),
                                      Mish(),
                                      )

    def forward(self, P5, P4, P3):
        D5 = self.p52d5(P5)    # [B, 512, 13, 13]
        D5_ = self.d52d5_(D5)  # [B, 256, 26, 26]
        P4_ = self.p42p4_(P4)  # [B, 256, 26, 26]
        D4 = self.d5_p4_2d4(torch.cat([D5_, P4_], dim=1))   # [B, 256, 26, 26]
        D4_ = self.d42d4_(D4)                               # [B, 128, 52, 52]
        P3_ = self.p32p3_(P3)                               # [B, 128, 52, 52]
        D3 = self.d4_p3_2d3(torch.cat([D4_, P3_], dim=1))   # [B, 128, 52, 52]

        U3 = D3                                             # [B, 128, 52, 52]   V
        U3_ = self.u32u3_(U3)
        U4 = self.d4u3_2u4(torch.cat([D4, U3_], dim=1))     # [B, 256, 26, 26]   V
        U4_ = self.u42u4_(U4)                               # [B, 512, 13, 13]
        U5 = self.d5u4_2u5(torch.cat([D5, U4_], dim=1))     # [B, 512, 13, 13]   V

        return [U5, U4, U3]


class YOLOv4(nn.Module):
    def __init__(self, backbone, num_classes=80):
        super(YOLOv4, self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.SPP = SPPNet()
        self.PANet = PANet()

        self.pred_s = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(256),
                                    Mish(),
                                    nn.Conv2d(256, 3 * (1 + 4 + self.num_classes), 1, stride=1, padding=0))

        self.pred_m = nn.Sequential(nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(512),
                                    Mish(),
                                    nn.Conv2d(512, 3 * (1 + 4 + self.num_classes), 1, stride=1, padding=0))

        self.pred_l = nn.Sequential(nn.Conv2d(512, 1024, 3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(1024),
                                    Mish(),
                                    nn.Conv2d(1024, 3 * (1 + 4 + self.num_classes), 1, stride=1, padding=0))

        print("num_params : ", self.count_parameters())

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):

        P3 = x = self.backbone.features1(x)  # [B, 256, 52, 52]
        P4 = x = self.backbone.features2(x)  # [B, 512, 26, 26]
        P5 = x = self.backbone.features3(x)  # [B, 1024, 13, 13]

        P5 = self.SPP(P5)
        U5, U4, U3 = self.PANet(P5, P4, P3)

        p_l = self.pred_l(U5).permute(0, 2, 3, 1)  # B, 13, 13, 255
        p_m = self.pred_m(U4).permute(0, 2, 3, 1)  # B, 26, 26, 255
        p_s = self.pred_s(U3).permute(0, 2, 3, 1)  # B, 52, 52, 255

        return [p_l, p_m, p_s]


if __name__ == '__main__':
    img = torch.randn([1, 3, 416, 416]).to(device)
    model = YOLOv4(backbone=CSPDarknet53(pretrained=True)).to(device)
    p_l, p_m, p_s = model(img)

    print("large : ", p_l.size())
    print("medium : ", p_m.size())
    print("small : ", p_s.size())

    '''
    torch.Size([1, 52, 52, 255])
    torch.Size([1, 26, 26, 255])
    torch.Size([1, 13, 13, 255])
    '''