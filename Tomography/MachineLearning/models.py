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
    
def DefaultConvNet(H,W,n_channels,n_classes):
    return ConvNet(H,W,n_channels, n_classes,[24,40,35],5,nn.ELU,[120,80,40])


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
    
def ResNet18(num_classes):
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)

    # Modify the first convolution layer to accept 2-channel images
    model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    return model

def ResNet34(num_classes):
    model = torchvision.models.resnet34(weights=None, num_classes=num_classes)

    # Modify the first convolution layer to accept 2-channel images
    model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    return model

def ResNet50(num_classes):
    model = torchvision.models.resnet50(weights=None, num_classes=num_classes)

    # Modify the first convolution layer to accept 2-channel images
    model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    return model
    
def MobileNet(num_classes):
    model = torchvision.models.mobilenet_v3_small(weights=None, num_classes=num_classes)

    # Modify the first convolutional layer to accept 2 input channels
    model.features[0][0] = torch.nn.Conv2d(2, model.features[0][0].out_channels, 
                                        kernel_size=model.features[0][0].kernel_size, 
                                        stride=model.features[0][0].stride, 
                                        padding=model.features[0][0].padding, 
                                        bias=model.features[0][0].bias is not None)
    
    return model

def EfficientNetB0(num_classes):
    efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 
                                  'nvidia_efficientnet_b0', pretrained=False,trust_repo=True)
    efficientnet.stem.conv = torch.nn.Conv2d(2, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    efficientnet.classifier.fc = torch.nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    return efficientnet