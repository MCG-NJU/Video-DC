import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule, NonLocal3d, build_activation_layer
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, Sequential
from mmengine.model.weight_init import constant_init, kaiming_init
from mmengine.runner.checkpoint import _load_checkpoint, load_checkpoint
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from torch.nn.modules.utils import _ntuple, _triple

from mmaction.registry import MODELS

@MODELS.register_module()
class MyConv3d(BaseModule):
    def __init__(self,
                 pretrained: Optional[str] = None,
                 stage_blocks: Optional[Tuple] = None,
                 in_channels: int = 3,
                 in_size: Tuple[int, int] = (224, 224),
                 channels: Sequence[int] = (3, 64, 128, 128),
                 temporal_strides: Sequence[int] = (1, 2, 2),
                 spatial_strides: Sequence[int] = (2, 2, 2),
                 num_stages: int = 4,
                 base_channels: int = 64,
                 style: str = 'pytorch',
                 frozen_stages: int = -1,
                 inflate: Sequence[int] = (1, 1, 1, 1),
                 inflate_style: str = '3x1x1',
                 conv_cfg: Dict = dict(type='Conv3d'),
                 norm_type: str = 'BN3d',
                 act_cfg: Dict = dict(type='ReLU', inplace=True),
                 norm_eval: bool = False,
                 with_cp: bool = False,
                 non_local: Sequence[int] = (0, 0, 0, 0),
                 zero_init_residual: bool = True,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.channels = channels
        self.temporal_strides = temporal_strides
        self.spatial_strides = spatial_strides
        self.base_channels = base_channels
        self.style = style
        self.frozen_stages = frozen_stages
        self.stage_inflations = _ntuple(num_stages)(inflate)
        self.non_local_stages = _ntuple(num_stages)(non_local)
        self.inflate_style = inflate_style
        self.conv_cfg = conv_cfg
        self.norm_type = norm_type
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        self.layers = self._make_layers(3) 


    def _make_layers(self, net_depth):
        layers = []
        for i in range(net_depth):
            pool_i = (self.temporal_strides[i], self.spatial_strides[i], self.spatial_strides[i])
            layers.append(nn.Conv3d(self.channels[i], self.channels[i+1], kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3)))
            layers.append(nn.BatchNorm3d(self.channels[i+1]) if self.norm_type == 'BN3d' else nn.GroupNorm(self.channels[i+1], self.channels[i+1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool3d(kernel_size=pool_i, stride=pool_i))
        # self.layers = nn.Sequential(*layers)
        # in_channels = self.in_channels
        # shape_feat = [in_channels, 16, im_size[0], im_size[1]]
        # for d in range(net_depth):
        #     layers += [nn.Conv3d(in_channels, 64 if d==0 else net_width, kernel_size=(3,7,7), padding=(1,3,3), stride=(1,2,2))]
        #     shape_feat[2] //= 2
        #     shape_feat[3] //= 2
        #     shape_feat[0] = 64 if d==0 else net_width
        #     layers += [nn.BatchNorm3d(shape_feat[0], affine=True)]
        #     layers += [nn.ReLU(inplace=True)]
        #     in_channels = shape_feat[0]
        #     layers += [nn.AvgPool3d(kernel_size=2, stride=2 if d!=0 else 1)]
        #     if d!=0:
        #         shape_feat[1] //= 2
        #     shape_feat[2] //= 2
        #     shape_feat[3] //= 2
        return nn.Sequential(*layers)
    
    def _freeze_stages(self) -> None:
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        assert 0
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    @staticmethod
    def _init_weights(self, pretrained: Optional[str] = None) -> None:
        """Initiate the parameters either from existing checkpoint or from
        scratch.

        Args:
            pretrained (str | None): The path of the pretrained weight. Will
                override the original `pretrained` if set. The arg is added to
                be compatible with mmdet. Defaults to None.
        """
        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')

            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                self.inflate_weights(logger)
            else:
                # Directly load 3D model.
                load_checkpoint(
                    self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck3d):
                        constant_init(m.conv3.bn, 0)
                    elif isinstance(m, BasicBlock3d):
                        constant_init(m.conv2.bn, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def init_weights(self, pretrained: Optional[str] = None) -> None:
        """Initialize weights."""
        self._init_weights(self, pretrained)

    def forward(self, x: torch.Tensor) \
            -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor or tuple[torch.Tensor]: The feature of the input
            samples extracted by the backbone.
        """
        return self.layers(x) 
    