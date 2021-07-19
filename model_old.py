import torch
import torch.nn as nn
import os
import sys
import numpy as np
import torch.nn.functional as F
import wget
from utils import bar_custom

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import device


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


norm_name = {"bn": nn.BatchNorm2d}
activate_name = {
    "relu": nn.ReLU,
    "leaky": nn.LeakyReLU,
    "linear": nn.Identity(),
    "mish": Mish(),
}


class Convolutional(nn.Module):
    def __init__(
            self,
            filters_in,
            filters_out,
            kernel_size,
            stride=1,
            norm="bn",
            activate="mish",
    ):
        super(Convolutional, self).__init__()

        self.norm = norm
        self.activate = activate

        self.__conv = nn.Conv2d(
            in_channels=filters_in,
            out_channels=filters_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=not norm,
        )
        if norm:
            assert norm in norm_name.keys()
            if norm == "bn":
                self.__norm = norm_name[norm](num_features=filters_out)

        if activate:
            assert activate in activate_name.keys()
            if activate == "leaky":
                self.__activate = activate_name[activate](
                    negative_slope=0.1, inplace=True
                )
            if activate == "relu":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "mish":
                self.__activate = activate_name[activate]

    def forward(self, x):
        x = self.__conv(x)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)

        return x


class CSPBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            hidden_channels=None,
            residual_activation="linear",
    ):
        super(CSPBlock, self).__init__()

        if hidden_channels is None:
            hidden_channels = out_channels

        self.block = nn.Sequential(
            Convolutional(in_channels, hidden_channels, 1),
            Convolutional(hidden_channels, out_channels, 3),
        )

        self.activation = activate_name[residual_activation]
        self.attention = "None"  # FIXME 일단은 None!

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


class CSPFirstStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSPFirstStage, self).__init__()

        self.downsample_conv = Convolutional(
            in_channels, out_channels, 3, stride=2
        )

        self.split_conv0 = Convolutional(out_channels, out_channels, 1)
        self.split_conv1 = Convolutional(out_channels, out_channels, 1)

        self.blocks_conv = nn.Sequential(
            CSPBlock(out_channels, out_channels, in_channels),
            Convolutional(out_channels, out_channels, 1),
        )

        self.concat_conv = Convolutional(out_channels * 2, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)

        x1 = self.blocks_conv(x1)

        x = torch.cat([x1, x0], dim=1)
        x = self.concat_conv(x)

        return x


class CSPStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(CSPStage, self).__init__()

        self.downsample_conv = Convolutional(
            in_channels, out_channels, 3, stride=2
        )

        self.split_conv0 = Convolutional(out_channels, out_channels // 2, 1)
        self.split_conv1 = Convolutional(out_channels, out_channels // 2, 1)

        self.blocks_conv = nn.Sequential(
            *[
                CSPBlock(out_channels // 2, out_channels // 2)
                for _ in range(num_blocks)
            ],
            Convolutional(out_channels // 2, out_channels // 2, 1)
        )

        self.concat_conv = Convolutional(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)
        # print('\tx0 : {}'.format(x0.shape))
        x1 = self.split_conv1(x)

        x1 = self.blocks_conv(x1)
        # print('\tx1 : {}'.format(x1.shape))

        x = torch.cat([x0, x1], dim=1)
        x = self.concat_conv(x)

        return x


class CSPDarknet53(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 pretrained=True,
                 stem_channels=32,
                 feature_channels=[64, 128, 256, 512, 1024],
                 num_features=3,
                 weight_path="./pretrained/yolov4.weights"
                 ):
        super(CSPDarknet53, self).__init__()
        self.num_classes = num_classes
        self.stem_conv = Convolutional(3, stem_channels, 3)
        self.stages = nn.ModuleList(
            [
                CSPFirstStage(stem_channels, feature_channels[0]),
                CSPStage(feature_channels[0], feature_channels[1], 2),
                CSPStage(feature_channels[1], feature_channels[2], 8),
                CSPStage(feature_channels[2], feature_channels[3], 8),
                CSPStage(feature_channels[3], feature_channels[4], 4),
            ]
        )
        self.feature_channels = feature_channels
        self.num_features = num_features

        if pretrained:

            os.makedirs(os.path.dirname(weight_path), exist_ok=True)
            if not os.path.exists(os.path.join(weight_path)):
                yolov4_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
                wget.download(url=yolov4_url, out=os.path.dirname(weight_path), bar=bar_custom)
                print('')
            else:
                print("YOLOv4 weight already exist!")

            self.load_CSPDarknet_weights(weight_path)
        # else:
        #     print("Need Darknet Weights")
        #     exit()

        print("num_params : ", self.count_parameters())

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        x = self.stem_conv(x)

        features = []
        for stage in self.stages:
            x = stage(x)
            # print('x : {}'.format(x.shape))
            features.append(x)
        return features[-self.num_features:]

    def load_CSPDarknet_weights(self, weight_file):
        print("load darknet weights : ", weight_file)
        with open(weight_file, "rb") as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
        count = 0
        ptr = 0
        for m in self.modules():
            if isinstance(m, Convolutional):
                # only initing backbone conv's weights
                # if count == cutoff:
                #     break
                # count += 1

                conv_layer = m._Convolutional__conv
                if m.norm == "bn":
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = m._Convolutional__norm
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(
                        bn_layer.bias.data
                    )
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(
                        bn_layer.weight.data
                    )
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(
                        weights[ptr: ptr + num_b]
                    ).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(
                        weights[ptr: ptr + num_b]
                    ).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b

                    # print("loading weight {}".format(bn_layer))
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]
                    ).view_as(conv_layer.bias.data)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(
                    conv_layer.weight.data
                )
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w


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

        self.feature_transform3_ = nn.Sequential(
            nn.Conv2d(256, 128, 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )

        self.feature_transform4_ = nn.Sequential(
            nn.Conv2d(512, 256, 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )

        self.upsample5_4 = nn.Sequential(
            nn.Conv2d(512, 256, 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2)
        )

        self.upsample4_3 = nn.Sequential(
            nn.Conv2d(256, 128, 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2)
        )

        self.downsample3_4 = nn.Sequential(
            nn.Conv2d(128, 256, 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )

        self.downsample4_5 = nn.Sequential(
            nn.Conv2d(256, 512, 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
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


class SPPNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
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

        self.maxpool5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool9 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool13 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

    def forward(self, x):
        x = self.conv(x)  # torch.Size([1, 512, 16, 16])
        maxpool5 = self.maxpool5(x)
        maxpool9 = self.maxpool9(x)
        maxpool13 = self.maxpool13(x)
        x = torch.cat([x, maxpool5, maxpool9, maxpool13], dim=1)
        return x


class YOLOv4(nn.Module):
    def __init__(self, backbone, num_classes=80):
        super(YOLOv4, self).__init__()
        self.backbone = backbone
        feature_channels = backbone.feature_channels[-3:]  # [256, 512, 1024]
        self.SPP = SpatialPyramidPooling(feature_channels)
        # self.SPP_ = SPPNet()
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

    # CSPnet
    img = torch.randn([1, 3, 512, 512]).to(device)
    model = CSPDarknet53(pretrained=True).to(device)
    feature_channels = model.feature_channels[-3:]
    print('feature_channels : {}'.format(feature_channels))
    test = model(img)
    for i, f in enumerate(test):
        print('{}_{}'.format(i, f.shape))

    # YOLOv4
    img_size = 416
    img = torch.randn([1, 3, img_size, img_size]).to(device)
    model = YOLOv4(CSPDarknet53(pretrained=True)).to(device)

    p_l, p_m, p_s = model(img)

    print("large  : ", p_l.size())
    print("medium : ", p_m.size())
    print("small  : ", p_s.size())

    '''
    large  :  torch.Size([1, 13, 13, 255])
    medium :  torch.Size([1, 26, 26, 255])
    small  :  torch.Size([1, 52, 52, 255])
    '''