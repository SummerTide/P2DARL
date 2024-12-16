

import ever as er
import torch, os
import time
import numpy as np
from skimage import measure
from tqdm import tqdm
from PIL import Image
from tools.metric import DamagePixelMetric

er.registry.register_all()

def visualize(loc, dam):
    loc = Image.fromarray(loc)
    loc.putpalette([0, 0, 0,
                    255, 255, 255])
    loc = loc.convert('RGB')

    dam = Image.fromarray(dam)
    dam.putpalette([0, 0, 0,
                    255, 255, 255,
                    0, 255, 0,
                    248, 179, 101,
                    255, 0, 0])
    dam = dam.convert('RGB')

    return loc, dam
    
def evaluate(self, test_dataloader, config=None):
    torch.cuda.empty_cache()
    self.model.eval()
    loc_metric_op = er.metric.PixelMetric(2, self.model_dir, logger=self.logger)
    damage_metric_op = DamagePixelMetric(max(self.model.module.config.head.dam_head.num_classes, 2), self.model_dir,
                                             logger=self.logger)
    object_metric_op = DamagePixelMetric(max(self.model.module.config.head.dam_head.num_classes, 2), self.model_dir,
                                             logger=self.logger)

    vis_dir = os.path.join(self.model_dir, 'vis-{}'.format(self.checkpoint.global_step))
    os.makedirs(vis_dir, exist_ok=True)
    # eval
    with torch.no_grad():
        for idx, (img, d) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            imname = d['image_filename']
            # print(imname)
            gt = d['masks']
            img = img.to(torch.device('cuda'))

            loc_prob, dam_prob = self.model(img)
            loc_prob = loc_prob.sigmoid()
            dam_prob = dam_prob.softmax(dim=1)
            # print(loc_prob.shape)  # [2, 1, 1024, 1024]
            # print(dam_prob.shape)  # [2, 5, 1024, 1024]

            loc_pred = (loc_prob > 0.5).cpu().numpy().squeeze(1)
            dam_pred = dam_prob.argmax(dim=1).cpu().numpy()
            # print(loc_pred.shape)
            # print(dam_pred.shape)
            _, object_dam = object_based_infer(loc_pred, dam_pred)
            # visualize
            for loc_pred_i, object_dam_i, fname_i in zip(loc_pred, object_dam, imname):
                loc_pred_i, object_dam_i = visualize(loc_pred_i.astype(np.uint8), object_dam_i.astype(np.uint8))
                fname_i = fname_i.replace('pre', 'dam')
                object_dam_i.save(os.path.join(vis_dir, fname_i))

            loc_pred = (loc_prob > 0.5).cpu().numpy().squeeze(1).ravel()
            loc_true = gt[:, :, :, 0].numpy().ravel()
            loc_true = np.where(loc_true > 0, np.ones_like(loc_true), np.zeros_like(loc_true))
            loc_metric_op.forward(loc_true, loc_pred)

            dam_pred = dam_prob.argmax(dim=1).cpu().numpy().ravel()
            dam_true = gt[:, :, :, 1].numpy().ravel()
            valid_inds = np.where(dam_true != 255)[0]
            dam_true = dam_true[valid_inds]
            dam_pred = dam_pred[valid_inds]
            damage_metric_op.forward(dam_true, dam_pred)

            object_dam = object_dam.ravel()
            object_dam = object_dam[valid_inds]
            object_metric_op.forward(dam_true, object_dam)


    loc_metric_op.summary_all()
    damage_metric_op.summary_all()
    object_metric_op.summary_all()

    torch.cuda.empty_cache()


def object_based_infer(pre_logit, post_logit):
    refined_dam = np.ones_like(post_logit) * 255
    for i, (single_loc, single_dam) in enumerate(zip(pre_logit, post_logit)):
        refined_dam[i, :, :] = _object_vote(single_loc, single_dam)

    return pre_logit, refined_dam


def _object_vote(loc, dam):
    # print(loc.shape)
    # print(dam.shape)
    damage_cls_list = [1, 2, 3, 4]
    local_mask = loc
    labeled_local, nums = measure.label(local_mask, connectivity=2, background=0, return_num=True)
    region_idlist = np.unique(labeled_local)
    if len(region_idlist) > 1:
        dam_mask = dam
        new_dam = local_mask.copy()
        for region_id in region_idlist:
            if all(local_mask[local_mask == region_id]) == 0:
                continue
            region_dam_count = [int(np.sum(dam_mask[labeled_local == region_id] == dam_cls_i)) * cls_weight \
                                # for dam_cls_i, cls_weight in zip(damage_cls_list, [8., 38., 25., 11.])]
                                for dam_cls_i, cls_weight in zip(damage_cls_list, [1., 1., 1., 1.])]
            dam_index = np.argmax(region_dam_count) + 1
            new_dam = np.where(labeled_local == region_id, dam_index, new_dam)
    else:
        new_dam = local_mask.copy()
    return new_dam


def register_evaluate_fn(launcher):
    launcher.override_evaluate(evaluate)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    trainer = er.trainer.get_trainer('th_amp_ddp')()
    trainer.run(after_construct_launcher_callbacks=[register_evaluate_fn])
