import torch
from torch import nn
import torch.nn.init as init
from math import prod
import torchvision


class LeNet(nn.Module):
    def __init__(self, H, W, input_channels, nclasses):
        out_conv_size = (H // 4 - 3) * (W // 4 - 3) * 16
        super(LeNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=6,
                      kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16,
                      kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(out_conv_size, 120),  # in_features = 16 x5x5
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, nclasses)
        )

    def forward(self, x):
        a1 = self.feature_extractor(x)
        a1 = torch.flatten(a1, 1)
        a2 = self.classifier(a1)
        return a2


def convolutional_block(in_channels, out_channels, kernel_size, activation):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=kernel_size),
        activation(),
        nn.AvgPool2d(kernel_size=2, stride=2),
    )


def fully_connected_block(in_features, out_features, activation):
    return nn.Sequential(
        nn.Linear(in_features=in_features, out_features=out_features),
        activation(),
    )


def out_res(n_blocks, H, W):
    for _ in range(n_blocks):
        H = H // 2 - 2
        W = W // 2 - 2
    return H, W


class Flatten1(nn.Module):
    def forward(self, input):
        return nn.Flatten(input, 1)


class ConvNet(nn.Module):
    def __init__(self, H, W, input_channels, nclasses, out_channels, kernel_size, activation, out_connected_layers):
        super(ConvNet, self).__init__()

        out_conv_size = prod(
            out_res(len(out_channels), H, W)) * out_channels[-1]

        in_channels = [input_channels] + out_channels[:-1]

        layers = []
        for _in_channels, _out_channels in zip(in_channels, out_channels):
            layers.append(convolutional_block(
                _in_channels, _out_channels, kernel_size, activation))

        self.conv_layers = nn.Sequential(*layers)

        in_size_connected_layers = [
            out_conv_size] + out_connected_layers[:-1]
        layers = []
        for _in_size, _out_size in zip(in_size_connected_layers, out_connected_layers):
            layers.append(
                nn.Linear(in_features=_in_size, out_features=_out_size))
            layers.append(activation())

        layers.append(
            nn.Linear(in_features=out_connected_layers[-1], out_features=nclasses))

        self.fully_connected_layers = nn.Sequential(*layers)

    def forward(self, x):
        result = self.conv_layers(x)
        result = torch.flatten(result, 1)
        result = self.fully_connected_layers(result)
        return result


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(SqueezeNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(512, 64, 256, 256),
        )

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


class AlexNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def my_convnext_tiny(*args, **kwargs):
    """ConvNeXt Tiny model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        weights (:class:`~torchvision.models.convnext.ConvNeXt_Tiny_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Tiny_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ConvNeXt_Tiny_Weights
        :members:
    """

    block_setting = [
        torchvision.models.convnext.CNBlockConfig(96, 192, 2),
        torchvision.models.convnext.CNBlockConfig(192, 384, 2),
        torchvision.models.convnext.CNBlockConfig(384, 768, 4),
        torchvision.models.convnext.CNBlockConfig(768, None, 2),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    return torchvision.models.convnext._convnext(block_setting, stochastic_depth_prob, None, False, **kwargs)


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


"""class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)),
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out"""


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 16)
        self.conv2 = conv_block(16, 32, pool=True)
        self.res1 = nn.Sequential(conv_block(32, 32), conv_block(32, 32))

        self.conv3 = conv_block(32, 64, pool=True)
        self.conv4 = conv_block(64, 128, pool=True)
        self.res2 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)),
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(128, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
