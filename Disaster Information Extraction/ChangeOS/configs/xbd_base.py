from albumentations import Compose, OneOf, Normalize
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomCrop
from ever.api.preprocess import albu
from albumentations.pytorch import ToTensorV2

data=dict(
    train=dict(
        type='Xview2PairwiseDataLoader',
        params=dict(
            image_dir=("/data1/gjc23/Dataset/xBD/train/images/", "/data1/gjc23/Dataset/xBD/tier3/images/"),
            label_dir=("/data1/gjc23/Dataset/xBD/train/labels/", "/data1/gjc23/Dataset/xBD/tier3/labels/"),
            mode='segm',
            include=('pre', 'post'),
            CV=dict(
                on=False,
                cur_k=0,
                k_fold=5,
            ),
            transforms=Compose([
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True)
                ], p=0.75),
                albu.RandomDiscreteScale(scales=[0.75, 1.25, 1.5], p=0.5),
                RandomCrop(640, 640, True),
                Normalize(mean=(0.485, 0.456, 0.406,
                                0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225,
                               0.229, 0.224, 0.225), max_pixel_value=255),
                ToTensorV2(True),
            ]),
            batch_size=16,
            num_workers=4,
            training=True
        ),
    ),
    test=dict(
        type='Xview2PairwiseDataLoader',
        params=dict(
            image_dir="/data1/gjc23/Dataset/xBD/hold/images/",
            label_dir="/data1/gjc23/Dataset/xBD/hold/labels/",
            # image_dir="/data1/gjc23/Dataset/hold/images/",
            # label_dir="/data1/gjc23/Dataset/hold/labels/",
            mode='segm',
            include=('pre', 'post'),
            CV=dict(
                on=False,
                cur_k=0,
                k_fold=5,
            ),
            transforms=Compose([
                Normalize(mean=(0.485, 0.456, 0.406,
                                0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225,
                               0.229, 0.224, 0.225), max_pixel_value=255),
                ToTensorV2(True),
            ]),
            batch_size=1,
            num_workers=4,
            training=False
        ),
    ),
)
optimizer=dict(
    type='sgd',
    params=dict(
        momentum=0.9,
        weight_decay=0.0001
    ),
    grad_clip=dict(
        max_norm=35,
        norm_type=2,
    )
)
learning_rate=dict(
    type='poly',
    params=dict(
        base_lr=0.03,
        power=0.9,
        max_iters=60000,
    ))
train=dict(
    forward_times=1,
    num_iters=60000,
    eval_per_epoch=True,
    summary_grads=False,
    summary_weights=False,
    distributed=True,
    apex_sync_bn=True,
    sync_bn=True,
    eval_after_train=True,
    log_interval_step=50,
    save_ckpt_interval_epoch=40,
    eval_interval_epoch=40,
)
test=dict(
)