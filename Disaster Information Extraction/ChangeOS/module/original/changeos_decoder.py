import ever as er
import ever.module as M
import torch
import torch.nn as nn


class FuseConv(nn.Sequential):
    def __init__(self, inchannels, outchannels):
        super(FuseConv, self).__init__(nn.Conv2d(inchannels, outchannels, kernel_size=1),
                                       nn.BatchNorm2d(outchannels),
                                       )
        self.relu = nn.ReLU(True)
        self.se = M.SEBlock(outchannels, 16)

    def forward(self, x):
        out = super(FuseConv, self).forward(x)
        residual = out
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class FuseMLP(nn.Sequential):
    def __init__(self, inchannels, outchannels):
        super().__init__(
            nn.Conv2d(inchannels, outchannels, kernel_size=1),
            LayerNorm2d(outchannels),
            nn.GELU(),
            nn.Conv2d(outchannels, outchannels, kernel_size=1),
            LayerNorm2d(outchannels),
        )


@er.registry.MODEL.register()
class ChangeOSDecoder(er.ERModule):
    def __init__(self, config):
        super().__init__(config)

        self.loc_neck = nn.Sequential(
            M.FPN(self.config.in_channels_list, self.config.out_channels, M.fpn.conv_bn_block),
            M.AssymetricDecoder(self.config.out_channels, self.config.out_channels)
        )

        self.dam_neck = nn.Sequential(
            M.FPN(self.config.in_channels_list, self.config.out_channels, M.fpn.conv_bn_block),
            M.AssymetricDecoder(self.config.out_channels, self.config.out_channels)
        )
        if self.config.fusion_type == 'residual_se':
            self.fuse_conv = FuseConv(2 * self.config.out_channels, self.config.out_channels)
        elif self.config.fusion_type == '2mlps':
            self.fuse_conv = FuseMLP(2 * self.config.out_channels, self.config.out_channels)
        else:
            raise ValueError(f'unknown fusion_type: {self.config.fusion_type}')

    def forward(self, t1_features, t2_features):
        t1_features = self.loc_neck(t1_features)
        t2_features = self.dam_neck(t2_features)

        st_features = self.fuse_conv(torch.cat([t1_features, t2_features], dim=1))
        return t1_features, st_features

    def set_default_config(self):
        self.config.update(dict(
            in_channels_list=(64, 128, 256, 512),
            out_channels=256,
            fusion_type='residual_se'
        ))
