from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models.resnet import conv1x1, BasicBlock, Bottleneck, model_urls
from torchvision.models.utils import load_state_dict_from_url

from models.base_model import BaseModel
from copy import deepcopy


class ResNet(BaseModel):
    def __init__(self,
            block: nn.Module,
            layers: List[int],
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: int = None,
            freeze_backbone = False,
            mean: Optional[Tuple[float, float, float]] = (0.485, 0.456, 0.406),
            std: Optional[Tuple[float, float, float]] = (0.229, 0.224, 0.225),
            **kwargs) -> None:
        self.backbone_features = 512 * block.expansion
        super(ResNet, self).__init__(**kwargs)
        self._norm_layer = nn.BatchNorm2d

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        # self.local_branch = self.create_local_branch(block, 512, layers[3], stride=1, dilate=False)
        # self.local_branch = deepcopy(self.layer4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Zero-init
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)
        
        ############################################################
        if freeze_backbone:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            for param in self.layer1.parameters():
                param.requires_grad = False
            for param in self.layer2.parameters():
                param.requires_grad = False
            for param in self.layer3.parameters():
                param.requires_grad = False
            for param in self.layer4.parameters():
                param.requires_grad = False
            for param in self.remap.parameters():
                param.requires_grad = False
            for param in self.norm_layer.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = False
        
            # for param in self.local_branch.parameters():
            #     param.requires_grad = False
        ###########################################################

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    # def create_local_branch(self, block, planes, blocks, stride=1, dilate=False):
    #     input_planes = 1024
    #     norm_layer = self._norm_layer
    #     downsample = None
    #     previous_dilation = self.dilation
    #     if dilate:
    #         self.dilation *= stride
    #         stride = 1
    #     if stride != 1 or input_planes != planes * 4:
    #         downsample = nn.Sequential(
    #             conv1x1(input_planes, planes * 4, stride),
    #             norm_layer(planes * 4),
    #         )

    #     layers = []
    #     layers.append(block(input_planes, planes, stride, downsample, self.groups,
    #                         self.base_width, previous_dilation, norm_layer))
    #     input_planes = planes * 4
    #     for _ in range(1, blocks):
    #         layers.append(block(input_planes, planes, groups=self.groups,
    #                             base_width=self.base_width, dilation=self.dilation,
    #                             norm_layer=norm_layer))

    #     return nn.Sequential(*layers)

    def feature_extractor(self, x: torch.Tensor):
        x = (x - self.mean) / self.std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # l = self.local_branch(x.detach())
        l = self.layer4(x)
        g = self.avgpool(l)
        g = torch.flatten(g, 1)

        # x = self.layer4(x)
        # x = self.avgpool(x)
        # features = torch.flatten(x, 1)
        # return features
        return g, l


def _resnet(arch: str, block: nn.Module, layers: List[int], pretrained: bool, progress: bool, **kwargs) -> nn.Module:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs) -> nn.Module:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

