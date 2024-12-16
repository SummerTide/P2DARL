import ever as er
import ever.module as M
import torch.nn.functional as F
from einops import rearrange
# from module.encoder.ops import FSRelationV2


@er.registry.MODEL.register()
class SiameseResNetEncoder(M.ResNetEncoder):
    def forward(self, inputs):
        x = rearrange(inputs, 'b (t c) h w -> (b t) c h w', t=2)
        bi_features = super().forward(x)
        if self.config.output_format == '2-4d':
            t1_features = []
            t2_features = []
            for bi_feat in bi_features:
                t1, t2 = rearrange(bi_feat, '(b t) c h w -> t b c h w', t=2)
                t1_features.append(t1)
                t2_features.append(t2)

            return t1_features, t2_features
        elif self.config.output_format == '5d':
            return [rearrange(bi_feat, '(b t) c h w -> b c t h w', t=2) for bi_feat in bi_features]

    def set_default_config(self):
        super().set_default_config()
        self.config.update(dict(
            output_format='2-4d'  # 2-4d, 5d
        ))


@er.registry.MODEL.register()
class SiameseResNetFPNEncoder(M.ResNetEncoder):
    def __init__(self, config):
        super().__init__(config)
        if self.config.resnet_type in ['resnet18', 'resnet34']:
            max_channels = 512
        else:
            max_channels = 2048
        self.fpn = M.FPN([max_channels // (2 ** (3 - i)) for i in range(4)], self.config.fpn_out_channels)

    def forward(self, inputs):
        x = rearrange(inputs, 'b (t c) h w -> (b t) c h w', t=2)
        bi_features = super().forward(x)
        bi_features = self.fpn(bi_features)

        if self.config.output_format == '2-4d':
            t1_features = []
            t2_features = []
            for bi_feat in bi_features:
                t1, t2 = rearrange(bi_feat, '(b t) c h w -> t b c h w', t=2)
                t1_features.append(t1)
                t2_features.append(t2)
            return t1_features, t2_features
        elif self.config.output_format == '5d':
            return [rearrange(bi_feat, '(b t) c h w -> b c t h w', t=2) for bi_feat in bi_features]

    def set_default_config(self):
        super().set_default_config()
        self.config.update(dict(
            output_format='2-4d',  # 2-4d, 5d
            fpn_out_channels=96,
        ))


@er.registry.MODEL.register()
class SiameseResNetFPNFSREncoder(M.ResNetEncoder):
    def __init__(self, config):
        super().__init__(config)
        if self.config.resnet_type in ['resnet18', 'resnet34']:
            max_channels = 512
        else:
            max_channels = 2048
        self.fpn = M.FPN([max_channels // (2 ** (3 - i)) for i in range(4)], self.config.fpn_out_channels)
        self.fsr = M.FSRelation(max_channels,
                                [self.config.fpn_out_channels for _ in range(4)],
                                self.config.fpn_out_channels,
                                True)

    def forward(self, inputs):
        x = rearrange(inputs, 'b (t c) h w -> (b t) c h w', t=2)
        bi_features = super().forward(x)
        coarsest_features = bi_features[-1]
        scene_embedding = F.adaptive_avg_pool2d(coarsest_features, 1)
        bi_features = self.fpn(bi_features)
        bi_features = self.fsr(scene_embedding, bi_features)

        if self.config.output_format == '2-4d':
            t1_features = []
            t2_features = []
            for bi_feat in bi_features:
                t1, t2 = rearrange(bi_feat, '(b t) c h w -> t b c h w', t=2)
                t1_features.append(t1)
                t2_features.append(t2)
            return t1_features, t2_features
        elif self.config.output_format == '5d':
            return [rearrange(bi_feat, '(b t) c h w -> b c t h w', t=2) for bi_feat in bi_features]

    def set_default_config(self):
        super().set_default_config()
        self.config.update(dict(
            output_format='2-4d',  # 2-4d, 5d
            fpn_out_channels=96,
        ))


# @er.registry.MODEL.register()
# class SiameseResNetFPNFSRV2Encoder(M.ResNetEncoder):
#     def __init__(self, config):
#         super().__init__(config)
#         if self.config.resnet_type in ['resnet18', 'resnet34']:
#             max_channels = 512
#         else:
#             max_channels = 2048
#         self.fpn = M.FPN([max_channels // (2 ** (3 - i)) for i in range(4)], self.config.fpn_out_channels)
#         self.fsr = FSRelationV2(max_channels,
#                                 [self.config.fpn_out_channels for _ in range(4)],
#                                 self.config.fpn_out_channels,
#                                 True)
#
#     def forward(self, inputs):
#         x = rearrange(inputs, 'b (t c) h w -> (b t) c h w', t=2)
#         bi_features = super().forward(x)
#         coarsest_features = bi_features[-1]
#         scene_embedding = F.adaptive_avg_pool2d(coarsest_features, 1)
#         bi_features = self.fpn(bi_features)
#         bi_features = self.fsr(scene_embedding, bi_features)
#
#         if self.config.output_format == '2-4d':
#             t1_features = []
#             t2_features = []
#             for bi_feat in bi_features:
#                 t1, t2 = rearrange(bi_feat, '(b t) c h w -> t b c h w', t=2)
#                 t1_features.append(t1)
#                 t2_features.append(t2)
#             return t1_features, t2_features
#         elif self.config.output_format == '5d':
#             return [rearrange(bi_feat, '(b t) c h w -> b c t h w', t=2) for bi_feat in bi_features]
#
#     def set_default_config(self):
#         super().set_default_config()
#         self.config.update(dict(
#             output_format='2-4d',  # 2-4d, 5d
#             fpn_out_channels=96,
#         ))


@er.registry.MODEL.register()
class DeepLabV3Encoder(M.ResNetEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        if self.config.resnet_type in ['resnet18', 'resnet34']:
            max_channels = 512
        else:
            max_channels = 2048

        self.h = M.AtrousSpatialPyramidPool(max_channels, self.cfg.out_channels, (12, 24, 36))

    def forward(self, inputs):
        features = super().forward(inputs)
        return self.h(features[-1])

    def set_default_config(self):
        super().set_default_config()
        self.config.update(dict(
            out_channels=-1
        ))


@er.registry.MODEL.register()
class PSPNetEncoder(M.ResNetEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        if self.config.resnet_type in ['resnet18', 'resnet34']:
            max_channels = 512
        else:
            max_channels = 2048

        self.h = M.PyramidPoolModule(max_channels, max_channels // 4, self.cfg.out_channels, (1, 2, 3, 4), '1x1', 0.1)

    def forward(self, inputs):
        features = super().forward(inputs)
        return self.h(features[-1])

    def set_default_config(self):
        super().set_default_config()
        self.config.update(dict(
            out_channels=-1
        ))
