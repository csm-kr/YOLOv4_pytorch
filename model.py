import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.CSPDarknet53 import CSPDarknet53
from config import device


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class SpatialPyramidPooling(nn.Module):
    def __init__(self, feature_channels, pool_sizes=[5, 9, 13]):  # feature channels = [256, 512, 1024]
        super(SpatialPyramidPooling, self).__init__()

        # head conv
        self.head_conv = nn.Sequential(
            Conv(feature_channels[-1], feature_channels[-1] // 2, 1),
            Conv(feature_channels[-1] // 2, feature_channels[-1], 3),
            Conv(feature_channels[-1], feature_channels[-1] // 2, 1),
        )

        self.maxpools = nn.ModuleList(
            [
                nn.MaxPool2d(pool_size, 1, pool_size // 2)
                for pool_size in pool_sizes
            ]
        )
        self.__initialize_weights()

    def forward(self, x):
        x = self.head_conv(x)  # torch.Size([1, 512, 16, 16])
        features = [maxpool(x) for maxpool in self.maxpools]  # torch.Size([1, 512, 16, 16])
        features = torch.cat([x] + features, dim=1)
        return features

    def __initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1), nn.Upsample(scale_factor=scale)
        )

    def forward(self, x):
        return self.upsample(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(Downsample, self).__init__()

        self.downsample = Conv(in_channels, out_channels, 3, 2)

    def forward(self, x):
        return self.downsample(x)


class PANet(nn.Module):
    def __init__(self, feature_channels):
        super(PANet, self).__init__()

        self.feature_transform3 = Conv(
            feature_channels[0], feature_channels[0] // 2, 1
        )
        self.feature_transform4 = Conv(
            feature_channels[1], feature_channels[1] // 2, 1
        )
        self.resample5_4 = Upsample(
            feature_channels[2] // 2, feature_channels[1] // 2
        )
        self.resample4_3 = Upsample(
            feature_channels[1] // 2, feature_channels[0] // 2
        )
        self.resample3_4 = Downsample(
            feature_channels[0] // 2, feature_channels[1] // 2
        )
        self.resample4_5 = Downsample(
            feature_channels[1] // 2, feature_channels[2] // 2
        )
        self.downstream_conv5 = nn.Sequential(
            Conv(feature_channels[2] * 2, feature_channels[2] // 2, 1),
            Conv(feature_channels[2] // 2, feature_channels[2], 3),
            Conv(feature_channels[2], feature_channels[2] // 2, 1),
        )
        self.downstream_conv4 = nn.Sequential(
            Conv(feature_channels[1], feature_channels[1] // 2, 1),
            Conv(feature_channels[1] // 2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1] // 2, 1),
            Conv(feature_channels[1] // 2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1] // 2, 1),
        )
        self.downstream_conv3 = nn.Sequential(
            Conv(feature_channels[0], feature_channels[0] // 2, 1),
            Conv(feature_channels[0] // 2, feature_channels[0], 3),
            Conv(feature_channels[0], feature_channels[0] // 2, 1),
            Conv(feature_channels[0] // 2, feature_channels[0], 3),
            Conv(feature_channels[0], feature_channels[0] // 2, 1),
        )
        self.upstream_conv4 = nn.Sequential(
            Conv(feature_channels[1], feature_channels[1] // 2, 1),
            Conv(feature_channels[1] // 2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1] // 2, 1),
            Conv(feature_channels[1] // 2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1] // 2, 1),
        )
        self.upstream_conv5 = nn.Sequential(
            Conv(feature_channels[2], feature_channels[2] // 2, 1),
            Conv(feature_channels[2] // 2, feature_channels[2], 3),
            Conv(feature_channels[2], feature_channels[2] // 2, 1),
            Conv(feature_channels[2] // 2, feature_channels[2], 3),
            Conv(feature_channels[2], feature_channels[2] // 2, 1),
        )
        self.__initialize_weights()

    def forward(self, features):
        features = [
            self.feature_transform3(features[0]),
            self.feature_transform4(features[1]),
            features[2],
        ]
        downstream_feature5 = self.downstream_conv5(features[2])
        downstream_feature4 = self.downstream_conv4(
            torch.cat(
                [features[1], self.resample5_4(downstream_feature5)], dim=1
            )
        )
        downstream_feature3 = self.downstream_conv3(
            torch.cat(
                [features[0], self.resample4_3(downstream_feature4)], dim=1
            )
        )
        upstream_feature4 = self.upstream_conv4(
            torch.cat(
                [self.resample3_4(downstream_feature3), downstream_feature4],
                dim=1,
            )
        )
        upstream_feature5 = self.upstream_conv5(
            torch.cat(
                [self.resample4_5(upstream_feature4), downstream_feature5],
                dim=1,
            )
        )
        return [downstream_feature3, upstream_feature4, upstream_feature5]

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class PredictNet(nn.Module):
    def __init__(self, feature_channels, target_channels):  # feature channels = [256, 512, 1024]
        super(PredictNet, self).__init__()

        self.predict_conv = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(feature_channels[i] // 2, feature_channels[i], 3),
                    nn.Conv2d(feature_channels[i], target_channels, 1),
                )
                for i in range(len(feature_channels))
            ]
        )
        self.__initialize_weights()

    def forward(self, features):
        predicts = [
            predict_conv(feature)
            for predict_conv, feature in zip(self.predict_conv, features)
        ]

        return predicts

    def __initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class YoloV4(nn.Module):
    def __init__(self, backbone, num_classes=80):
        super(YoloV4, self).__init__()
        self.backbone = backbone
        feature_channels = backbone.feature_channels[-3:]  # [256, 512, 1024]
        self.SPP = SpatialPyramidPooling(feature_channels)
        self.PANet = PANet(feature_channels)
        self.predict_net = PredictNet(feature_channels, target_channels=3 * (num_classes + 5))
        print("num_params : ", self.count_parameters())

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        features = self.backbone(x)
        features[-1] = self.SPP(features[-1])
        features = self.PANet(features)
        predicts = self.predict_net(features)
        p1, p2, p3 = predicts

        p_s = p1.permute(0, 2, 3, 1)  # B, 52, 52, 255
        p_m = p2.permute(0, 2, 3, 1)  # B, 26, 26, 255
        p_l = p3.permute(0, 2, 3, 1)  # B, 13, 13, 255

        return [p_l, p_m, p_s]


if __name__ == "__main__":
    img_size = 416
    img = torch.randn([1, 3, img_size, img_size]).to(device)
    model = YoloV4(CSPDarknet53(pretrained=True)).to(device)

    p_l, p_m, p_s = model(img)

    print("large  : ", p_l.size())
    print("medium : ", p_m.size())
    print("small  : ", p_s.size())

    '''
    large  :  torch.Size([1, 13, 13, 255])
    medium :  torch.Size([1, 26, 26, 255])
    small  :  torch.Size([1, 52, 52, 255])
    '''