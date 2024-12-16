from module.original.changeos_decoder import ChangeOSDecoder
from module.original.changeos_head import ChangeOSHead
import ever as er
import torch

@er.registry.MODEL.register()
class ChangeOS(er.ERModule):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = er.registry.MODEL[self.config.encoder.type](self.config.encoder.params)
        self.decoder = ChangeOSDecoder(self.config.decoder)
        self.head = ChangeOSHead(self.config.head)
        self.init_from_weight_file()

    def forward(self, x, y=None):
        features = self.encoder(x)
        decoder_features = self.decoder(*features)
        t1_features, st_features = decoder_features

        if self.training:
            return self.head(t1_features, st_features, y)
        else:
            loc_logit = self.head.loc_cls(t1_features)
            dam_logit = self.head.dam_cls(st_features)
            return loc_logit, dam_logit

    def set_default_config(self):
        self.config.update(dict(
            encoder_bitemporal_forward=False,
            encoder=dict(
                resnet_type='resnet18',
                pretrained=True,
                output_stride=32,
                output_format='2-4d',
            ),
            decoder=dict(
                in_channels_list=(64, 128, 256, 512),
                out_channels=256,
            ),
            head=dict(
                loc_head=dict(
                    in_channels=256,
                    bottlneck_channels=128,
                    num_blocks=1,
                    num_classes=1,
                    upsample_scale=4.
                ),
                dam_head=dict(
                    in_channels=256,
                    bottlneck_channels=128,
                    num_blocks=1,
                    num_classes=5,
                    upsample_scale=4.
                ),
                loss=dict(
                    loc=dict(
                        bce=dict(),
                        tver=dict(alpha=0.9),
                        log_binary_iou_sigmoid=dict(),
                        mem=dict(),
                        prefix='loc_',
                        ignore_index=255,
                    ),
                    dam=dict(
                        ce=dict(),
                        prefix='dam_',
                        ignore_index=255,
                    )
                )
            ),
        ))

if __name__ == '__main__':
    from module.original.resnet import SiameseResNetEncoder
    er.registry.register_all()
    x = torch.ones((2, 6, 1024, 1024))
    y = dict(
        masks=torch.ones((2, 2, 1024, 1024)).long(),
    )

    m = ChangeOS(dict(
        encoder=dict(
            type='SiameseResNetEncoder',
            params=dict(
                resnet_type='resnet18',
                pretrained=True,
                output_stride=32,
                output_format='2-4d',
            ),
        ),
        decoder=dict(
            type='ChangeOSDecoder',
            params=dict(
                in_channels_list=(64, 128, 256, 512),
                out_channels=256,
                fusion_type='2mlps'
            ),
        ),
        head=dict(
            type='ChangeOSHead',
            params=dict(
                loc_head=dict(
                    in_channels=256,
                    bottlneck_channels=128,
                    num_blocks=1,
                    num_classes=1,
                    upsample_scale=4.,
                    deep_head=False,
                ),
                dam_head=dict(
                    in_channels=256,
                    bottlneck_channels=128,
                    num_blocks=1,
                    num_classes=5,
                    upsample_scale=4.,
                    deep_head=False,
                ),
                loss=dict(
                    loc=dict(
                        bce=dict(),
                        tver=dict(alpha=0.9),
                        log_binary_iou_sigmoid=dict(),
                        mem=dict(),
                        prefix='loc_'
                    ),
                    dam=dict(
                        ce=dict(),
                        prefix='dam_'
                    )
                )
            )
        ),
    ))
    loss = m(x, y)
    print(loss)
