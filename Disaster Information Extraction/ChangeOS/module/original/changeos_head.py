import torch
import torch.nn as nn
import ever as er
import ever.module as M
import numpy as np
from skimage import measure
from .comm import MultiSegmentation


class DeepHead(nn.Module):
    def __init__(self, in_channels, bottlneck_channels, num_blocks, num_classes, upsample_scale):
        super(DeepHead, self).__init__()
        assert num_blocks > 0
        self.relu = nn.ReLU(True)
        self.blocks = nn.ModuleList([nn.Sequential(
            # 1x1
            nn.Conv2d(in_channels, bottlneck_channels, 1),
            nn.BatchNorm2d(bottlneck_channels),
            nn.ReLU(True),
            # 3x3
            nn.Conv2d(bottlneck_channels, bottlneck_channels, 3, 1, 1),
            nn.BatchNorm2d(bottlneck_channels),
            # 1x1
            nn.Conv2d(bottlneck_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            M.SEBlock(in_channels, 16)
        ) for _ in range(num_blocks)])

        self.cls = nn.Conv2d(in_channels, num_classes, 1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=upsample_scale)

    def forward(self, x, upsample=True):
        indentity = x
        for m in self.blocks:
            x = m(x)
            x += indentity
            x = self.relu(x)
            indentity = x
        x = self.cls(x)
        if upsample:
            x = self.up(x)
        return x


@er.registry.MODEL.register()
class ChangeOSHead(er.ERModule, MultiSegmentation):
    def __init__(self, config):
        super().__init__(config)
        if self.config.loc_head.deep_head:
            self.config.loc_head.pop('deep_head')
            self.loc_cls = DeepHead(**self.config.loc_head)
        else:
            self.loc_cls = M.ConvUpsampling(
                self.config.loc_head.in_channels,
                self.config.loc_head.num_classes,
                self.config.loc_head.upsample_scale,
                1
            )

        if self.config.dam_head.deep_head:
            self.config.dam_head.pop('deep_head')
            self.dam_cls = DeepHead(**self.config.dam_head)
        else:
            self.dam_cls = M.ConvUpsampling(
                self.config.dam_head.in_channels,
                self.config.dam_head.num_classes,
                self.config.dam_head.upsample_scale,
                1
            )

    def forward(self, t1_features, st_features, y=None):
        loc_logit = self.loc_cls(t1_features)
        dam_logit = self.dam_cls(st_features)

        if self.training:
            loss_dict = dict()
            gt_pre = (y['masks'][:, :, :, 0]).float()
            gt_post = y['masks'][:, :, :, 1].long()

            loss_dict.update(self.loss(gt_pre, loc_logit, self.config.loss.loc))
            loss_dict.update(self.loss(gt_post, dam_logit, self.config.loss.dam))

            return loss_dict

        if self.config.inference_mode == 'pixel-based':
            return self.pixel_based_infer(loc_logit, dam_logit)
        elif self.config.inference_mode == 'object-based':
            return self.object_based_infer(loc_logit, dam_logit)
        elif self.config.inference_mode == 'raw_cat':
            return torch.cat([loc_logit.sigmoid(), dam_logit.softmax(dim=1)], dim=1)

    def pixel_based_infer(self, pre_pred, post_pred, logit=True):
        if logit:
            pr_loc = pre_pred > 0.
        else:
            pr_loc = pre_pred > .5
        pr_dam = post_pred.argmax(dim=1, keepdim=True)
        return pr_loc, pr_dam

    def object_based_infer(self, pre_pred, post_pred, logit=True):
        if logit:
            loc = pre_pred > 0.
        else:
            loc = pre_pred > .5
        loc = loc.cpu().squeeze(1).numpy()
        dam = post_pred.argmax(dim=1).cpu().squeeze(1).numpy()

        refined_dam = np.zeros_like(dam)
        for i, (single_loc, single_dam) in enumerate(zip(loc, dam)):
            refined_dam[i, :, :] = _object_vote(single_loc, single_dam)

        return torch.from_numpy(loc), torch.from_numpy(refined_dam)

    def set_default_config(self):
        self.config.update(dict(
            loc_head=dict(
                in_channels=256,
                bottlneck_channels=128,
                num_blocks=1,
                num_classes=1,
                upsample_scale=4.,
                deep_head=True,
            ),
            dam_head=dict(
                in_channels=256,
                bottlneck_channels=128,
                num_blocks=1,
                num_classes=5,
                upsample_scale=4.,
                deep_head=True,
            ),
            loss=dict(
                loc=dict(
                    bce=dict(),
                    tver=dict(alpha=0.9),
                    log_binary_iou_sigmoid=dict(),
                    ignore_index=255
                ),
                dam=dict(
                    ce=dict(),
                    ignore_index=255
                )
            ),
            inference_mode='raw_cat'
        ))


def _object_vote(loc, dam, cls_weight_list=(8., 38., 25., 11.)):
    damage_cls_list = [1, 2, 3, 4]
    # 1. read localization mask
    local_mask = loc
    # 2. get connected regions
    labeled_local, nums = measure.label(local_mask, connectivity=2, background=0, return_num=True)
    region_idlist = np.unique(labeled_local)
    # 3. start vote
    if len(region_idlist) > 1:
        dam_mask = dam
        new_dam = local_mask.copy()
        for region_id in region_idlist:
            # if background, ignore it
            if all(local_mask[local_mask == region_id]) == 0:
                continue
            region_dam_count = [int(np.sum(dam_mask[labeled_local == region_id] == dam_cls_i)) * cls_weight \
                                for dam_cls_i, cls_weight in zip(damage_cls_list, cls_weight_list)]
            # vote
            dam_index = np.argmax(region_dam_count) + 1
            new_dam = np.where(labeled_local == region_id, dam_index, new_dam)
    else:
        new_dam = local_mask.copy()

    return new_dam
