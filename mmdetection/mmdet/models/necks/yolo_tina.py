# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2019 Western Digital Corporation or its affiliates.
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class DetectionBlock(BaseModule):
    """Detection block in YOLO neck.

    Let out_channels = n, the DetectionBlock contains:
    Six ConvLayers, 1 Conv2D Layer and 1 YoloLayer.
    The first 6 ConvLayers are formed the following way:
        1x1xn, 3x3x2n, 1x1xn, 3x3x2n, 1x1xn, 3x3x2n.
    The Conv2D layer is 1x1x255.
    Some block will have branch after the fifth ConvLayer.
    The input channel is arbitrary (in_channels)

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: ConfigType = dict(type='LeakyReLU', negative_slope=0.1),
                 init_cfg: OptMultiConfig = None) -> None:
        super(DetectionBlock, self).__init__(init_cfg)
        double_out_channels = out_channels * 2

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1 = ConvModule(in_channels, out_channels, 1, **cfg)
        self.conv2 = ConvModule(
            out_channels, double_out_channels, 3, padding=1, **cfg)
        self.conv3 = ConvModule(double_out_channels, out_channels, 1, **cfg)
        self.conv4 = ConvModule(
            out_channels, double_out_channels, 3, padding=1, **cfg)
        self.conv5 = ConvModule(double_out_channels, out_channels, 1, **cfg)

    def forward(self, x: Tensor) -> Tensor:
        tmp = self.conv1(x)
        tmp = self.conv2(tmp)
        tmp = self.conv3(tmp)
        tmp = self.conv4(tmp)
        out = self.conv5(tmp)
        return out

# class Inception(BaseModule):
#
#     def __init__(self,
#                  in_channel,
#                  conv_cfg=None,
#                  norm_cfg=None,
#                  dcn_cfg=None,
#                  share=False):
#         super(Inception, self).__init__()
#         assert in_channel % 2**2 == 0
#
#         self.in_channel = in_channel
#         self.conv_cfg = conv_cfg
#         self.norm_cfg = norm_cfg
#         assert dcn_cfg is None or isinstance(dcn_cfg, dict)
#         self.dcn_cfg = dcn_cfg
#         self.with_dcn = True if dcn_cfg is not None else False
#         self.share = share
#
#         self.convs = nn.ModuleList()
#
#         conv = ConvModule(
#             self.in_channel,
#             self.in_channel // 2,
#             3,
#             padding=1,
#             conv_cfg=self.conv_cfg,
#             norm_cfg=self.norm_cfg,
#             act_cfg=None)
#
#         self.convs.append(conv)
#
#         conv = ConvModule(
#             self.in_channel,
#             self.in_channel // 4,
#             3,
#             padding=1,
#             conv_cfg=self.conv_cfg,
#             norm_cfg=self.norm_cfg)
#
#         self.convs.append(conv)
#
#         for i in range(3):
#             act_cfg = dict(type ='ReLU') if i % 2 == 1 else None
#             conv = ConvModule(
#                 self.in_channel // 4,
#                 self.in_channel // 4,
#                 3,
#                 padding=1,
#                 conv_cfg=self.conv_cfg,
#                 norm_cfg=self.norm_cfg,
#                 act_cfg=act_cfg)
#
#             self.convs.append(conv)
#         self.convs.to('cuda')
#         if self.with_dcn:
#             self.dcn = ConvModule(
#                 self.in_channel,
#                 self.in_channel,
#                 3,
#                 padding=1,
#                 conv_cfg=self.dcn_cfg,
#                 norm_cfg=self.norm_cfg,
#                 act_cfg=act_cfg)
#
#     def init_weights(self):
#         """Initialize the weights of FPN module."""
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 xavier_init(m, distribution='uniform')
#
#     def forward(self, x):
#         if self.share:
#                 x_3 = self.level_convs[0](x)
#
#                 x_5_1 = self.level_convs[1](x)
#                 x_5 = self.level_convs[2](x_5_1)
#
#                 x_7_2 = self.level_convs[3](x_5_1)
#                 x_7 = self.level_convs[4](x_7_2)
#         else:
#             x_3 = self.convs[0](x)
#
#             x_5_1 = self.convs[1](x)
#             x_5 = self.convs[2](x_5_1)
#
#             x_7_2 = self.convs[3](x_5_1)
#             x_7 = self.convs[4](x_7_2)
#         out = F.relu(torch.cat([x_3, x_5, x_7], dim=1))
#
#         if self.with_dcn:
#             out = self.dcn(out)
#
#         return out

class InceptionDWConv2d(BaseModule):
    def __init__(self, in_channels, band_kernel_size, square_kernel_size=3, branch_ratio=1 / 3):
        super().__init__()
        self.group = 1 / branch_ratio
        self.gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.gc_Conv = int(self.gc * 0.5)
        self.gc_Identity = int(self.gc - self.gc_Conv)
        self.dwconv_hw = nn.Conv2d(self.gc_Conv, self.gc_Conv, square_kernel_size, padding=square_kernel_size // 2,
                                   groups=self.gc_Conv)
        self.dwconv_w = nn.Conv2d(self.gc_Conv, self.gc_Conv, kernel_size=(1, band_kernel_size),
                                  padding=(0, band_kernel_size // 2), groups=self.gc_Conv)
        self.dwconv_h = nn.Conv2d(self.gc_Conv, self.gc_Conv, kernel_size=(band_kernel_size, 1),
                                  padding=(band_kernel_size // 2, 0), groups=self.gc_Conv)
        self.split_indexes = (self.gc, self.gc, self.gc)
        self.dwconv_h = self.dwconv_h.to(device)
        self.dwconv_w = self.dwconv_w.to(device)
        self.dwconv_hw = self.dwconv_hw.to(device)


    def forward(self, x):
        # x = self.channel_shuffle(x)
        split_indexes_2 = (int(self.gc_Conv), int(self.gc_Identity))
        x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        x_hw_Conv, x_hw_Identity = torch.split(x_hw, split_indexes_2, dim=1)
        x_w_Conv, x_w_Identity = torch.split(x_w, split_indexes_2, dim=1)
        x_h_Conv, x_h_Identity = torch.split(x_h, split_indexes_2, dim=1)
        out = F.relu(torch.cat(
            (self.dwconv_hw(x_hw_Conv), x_hw_Identity, self.dwconv_w(x_w_Conv), x_w_Identity, self.dwconv_h(x_h_Conv),
             x_h_Identity),
            dim=1,
        ))
        return out

@MODELS.register_module()
class YOLOTina(BaseModule):
    """The neck of YOLOV3.

    It can be treated as a simplified version of FPN. It
    will take the result from Darknet backbone and do some upsampling and
    concatenation. It will finally output the detection result.

    Note:
        The input feats should be from top to bottom.
            i.e., from high-lvl to low-lvl
        But YOLOV3Neck will process them in reversed order.
            i.e., from bottom (high-lvl) to top (low-lvl)

    Args:
        num_scales (int): The number of scales / stages.
        in_channels (List[int]): The number of input channels per scale.
        out_channels (List[int]): The number of output channels  per scale.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Dictionary to construct and config norm
            layer. Default: dict(type='BN', requires_grad=True)
        act_cfg (dict, optional): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_scales: int,
                 band_kernel_size: int,
                 in_channels: List[int],
                 out_channels: List[int],
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: ConfigType = dict(
                     type='LeakyReLU', negative_slope=0.1),
                 init_cfg: OptMultiConfig = None) -> None:
        super(YOLOTina, self).__init__(init_cfg)
        assert (num_scales == len(in_channels) == len(out_channels))
        self.num_scales = num_scales
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.band_kernel_size = band_kernel_size

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # To support arbitrary scales, the code looks awful, but it works.
        # Better solution is welcomed.
        self.detect1 = DetectionBlock(in_channels[0], out_channels[0], **cfg)
        # 1, 2
        for i in range(1, self.num_scales):
            in_c, out_c = self.in_channels[i], self.out_channels[i]
            inter_c = out_channels[i - 1]
            self.add_module(f'conv{i}', ConvModule(inter_c, out_c, 1, **cfg))
            # in_c + out_c : High-lvl feats will be cat with low-lvl feats
            self.add_module(f'detect{i+1}',
                            DetectionBlock(in_c + out_c, out_c, **cfg))

    def forward(self, feats=Tuple[Tensor]) -> Tuple[Tensor]:
        assert len(feats) == self.num_scales

        # processed from bottom (high-lvl) to top (low-lvl)
        outs = []
        out = self.detect1(feats[-1])
        outs.append(out)

        for i, x in enumerate(reversed(feats[:-1])):
            conv = getattr(self, f'conv{i+1}')
            tmp = conv(out)

            # Cat with low-lvl feats
            tmp = F.interpolate(tmp, scale_factor=2)
            tmp = torch.cat((tmp, x), 1)

            detect = getattr(self, f'detect{i+2}')
            out = detect(tmp)
            outs.append(out)

        in_channel = [x.shape[1] for x in outs]
        output = []
        for i, x in enumerate(outs):
            # print(in_channel[i], x.shape)
            inception = InceptionDWConv2d(in_channels=in_channel[i] ,band_kernel_size=self.band_kernel_size)
            out = inception(x)
            output.append(out)

        return tuple(output)
