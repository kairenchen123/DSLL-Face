# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
import os
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
import torch.nn.functional as F
from mmdet.registry import MODELS
from ..layers import CSPLayer


class SimConv(nn.Module):
    '''Normal Conv with ReLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False):
        super(SimConv, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class SimSPPF(nn.Module):
    '''Simplified SPPF with ReLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(SimSPPF, self).__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = SimConv(in_channels, c_, 1, 1).to('cuda')
        self.cv2 = SimConv(c_ * 4, out_channels, 1, 1).to('cuda')
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2).to('cuda')# 保证feature的尺寸不变

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

# class SPPF_bf(BaseModule):
#     def __init__(self,
#                  in_channels,
#                  conv_cfg=None,
#                  norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
#                  act_cfg=dict(type='Swish'),
#                  init_cfg=None):
#         super(SPPF_bf, self).__init__(init_cfg)
#         self.in_channels = in_channels
#         self.spp_conv1_down=ConvModule(
#                     in_channels,
#                     in_channels//2,
#                     1,
#                     conv_cfg=conv_cfg,
#                     norm_cfg=norm_cfg,
#                     act_cfg=act_cfg
#                 )
#         self.spp_conv1_down =self.spp_conv1_down.to('cuda')
#         self.spp_conv2_down=ConvModule(
#                 in_channels//2,
#                 in_channels,
#                 3,
#                 1,
#                 1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg,
#                 act_cfg=act_cfg
#             )
#         self.spp_conv2_down = self.spp_conv2_down.to('cuda')
#         self.spp_conv3_down=ConvModule(
#                 in_channels,
#                 in_channels // 2,
#                 1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg,
#                 act_cfg=act_cfg
#             )
#         self.spp_conv3_down = self.spp_conv3_down.to('cuda')
#         self.spp_conv1_up=ConvModule(
#                 in_channels * 2,
#                 in_channels // 2,
#                 1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg,
#                 act_cfg=act_cfg
#             )
#         self.spp_conv1_up = self.spp_conv1_up.to('cuda')
#         self.spp_conv2_up=ConvModule(
#                 in_channels // 2,
#                 in_channels,
#                 3,
#                 1,
#                 1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg,
#                 act_cfg=act_cfg
#             )
#         self.spp_conv2_up = self.spp_conv2_up.to('cuda')
#     def forward(self, x):
#         x = self.spp_conv1_down(x)
#         x = self.spp_conv2_down(x)
#         x = self.spp_conv3_down(x)
#         simsppf = SimSPPF(in_channels=self.in_channels//2, out_channels=self.in_channels*2)
#         x = simsppf(x)
#         x = self.spp_conv1_up(x)
#         x = self.spp_conv2_up(x)
#
#         return x
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class InceptionDWConv2d(BaseModule):
    def __init__(self,
                 in_channels,
                 band_kernel_size,
                 square_kernel_size=3,
                 branch_ratio=1 / 3,
                 conv_cfg=None,
                 norm_cfg=None,
                 ):
        super().__init__()
        self.group = 1 / branch_ratio
        self.gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.gc_Conv = int(self.gc * 0.5)
        self.gc_Identity = int(self.gc - self.gc_Conv)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dwconv_hw = ConvModule(self.gc_Conv,
                                    self.gc_Conv,
                                    square_kernel_size,
                                    padding=square_kernel_size // 2,
                                    groups=self.gc_Conv,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg)
        self.dwconv_w = ConvModule(self.gc_Conv,
                                   self.gc_Conv,
                                   kernel_size=(1, band_kernel_size),
                                   padding=(0, band_kernel_size // 2),
                                   groups=self.gc_Conv,
                                   conv_cfg=self.conv_cfg,
                                   norm_cfg=self.norm_cfg)
        self.dwconv_h = ConvModule(self.gc_Conv,
                                   self.gc_Conv,
                                   kernel_size=(band_kernel_size, 1),
                                   padding=(band_kernel_size // 2, 0),
                                   groups=self.gc_Conv,
                                   conv_cfg=self.conv_cfg,
                                   norm_cfg=self.norm_cfg)
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
class YOLOX_INTERNIMAGE_PAFPN(BaseModule):
    """Path Aggregation Network used in YOLOX.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 3
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish')
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 band_kernel_size,
                 num_csp_blocks=3,
                 use_depthwise=False,
                 upsample_cfg=dict(scale_factor=2, mode='nearest'),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super(YOLOX_INTERNIMAGE_PAFPN, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.band_kernel_size = band_kernel_size
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv(
                    in_channels[idx],
                    in_channels[idx],
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: YOLOXPAFPN features.
        """
        assert len(inputs) == len(self.in_channels)
        # sim_sppf
        # spp_in = inputs[-1]
        # spp = SimSPPF(in_channels=self.in_channels[-1], out_channels=self.in_channels[-1])
        # x = spp(spp_in)
        # inputs[-1] = x

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        # inception
        inception_outs = []
        for idx in range(len(outs)):
            inception = InceptionDWConv2d(in_channels=self.out_channels, band_kernel_size=self.band_kernel_size, norm_cfg=self.norm_cfg)
            x = inception(outs[idx])
            inception_outs.append(x)

        return tuple(inception_outs)
