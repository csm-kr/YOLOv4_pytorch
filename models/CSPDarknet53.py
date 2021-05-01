import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from config import device
from models.attention_layers import SEModule, CBAM


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

        self.attention = "None"     #FIXME 일단은 None!
        if self.attention == "SEnet":
            self.attention_module = SEModule(out_channels)
        elif self.attention == "CBAM":
            self.attention_module = CBAM(out_channels)
        else:
            self.attention = None

    def forward(self, x):
        residual = x
        out = self.block(x)
        if self.attention is not None:
            out = self.attention_module(out)
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
        weight_path="./data/weights/yolov4.weights"
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
            self.load_CSPDarknet_weights(weight_path)
        # else:
        #     print("Need Darknet Weights")
        #     exit()
    
    def forward(self, x):
        x = self.stem_conv(x)
        
        features = []
        for stage in self.stages:
            x = stage(x)
            # print('x : {}'.format(x.shape))
            features.append(x)
        return features[-self.num_features :]

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
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.bias.data
                    )
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                        bn_layer.weight.data
                    )
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(
                        weights[ptr : ptr + num_b]
                    ).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(
                        weights[ptr : ptr + num_b]
                    ).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b

                    # print("loading weight {}".format(bn_layer))
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr : ptr + num_b]
                    ).view_as(conv_layer.bias.data)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(
                    conv_layer.weight.data
                )
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

                # print("loading weight {}".format(conv_layer))

# class YOLOv4(nn.Module):
#     def __init__(self):

#     def forward(self, x):


if __name__ == '__main__':
    img = torch.randn([1, 3, 512, 512]).to(device)
    
    model = CSPDarknet53(pretrained=True).to(device)
    feature_channels = model.feature_channels[-3:]
    print('feature_channels : {}'.format(feature_channels))
    test = model(img)
    for i, f in enumerate(test):
        print('{}_{}'.format(i, f.shape))

        