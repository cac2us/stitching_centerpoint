import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor

tasks = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=1, class_names=["truck"]),
    dict(num_class=1, class_names=["heavy_truck"]),
    dict(num_class=1, class_names=["bus"]),
    dict(num_class=1, class_names=["motorcycle"]),
    # dict(num_class=1, class_names=["bicycle"]),
    dict(num_class=1, class_names=["pedestrian"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

# model settings
model = dict(
    type="VoxelNet",
    pretrained=None,
    reader=dict(
        type="VoxelFeatureExtractorV3",
        num_input_features=3,
    ),
    backbone=dict(
        type="SpMiddleResNetFHD", num_input_features=3, ds_factor=8
    ),
    ## original
    # neck=dict(
    #     type="RPN",
    #     layer_nums=[5, 5],
    #     ds_layer_strides=[1, 2],
    #     ds_num_filters=[128, 256],
    #     us_layer_strides=[1, 2],
    #     us_num_filters=[256, 256],
    #     num_input_features=256,
    #     logger=logging.getLogger("RPN"),
    # ),
    neck=dict(
        type="RPN",
        layer_nums=[5, 5],
        ds_layer_strides=[1, 2],
        ds_num_filters=[256, 512],
        us_layer_strides=[1, 2],
        us_num_filters=[256, 512],
        num_input_features=256,
        logger=logging.getLogger("RPN"),
    ),

    bbox_head=dict(
        type="CenterHead",
        in_channels=sum([256, 256]),
        tasks=tasks,
        dataset='stitch_dataset',
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2), 'vel': (2, 2)}, # (output_channel, num_conv)
        share_conv_channel=64,
        dcn_head=False 
    ),
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
)


train_cfg = dict(assigner=assigner)

test_cfg = dict(
    post_center_limit_range = [-6.4, -16.5, -5.6, 80.0, 62.7, -1.3],
    nms=dict(
        nms_pre_max_size=1000,
        nms_post_max_size=83,
        nms_iou_threshold=0.001,
    ),
    score_threshold=0.1,
    pc_range=[208413.9, 534700.9],
    out_size_factor=get_downsample_factor(model),
    voxel_size=[0.1, 0.1]
)


# dataset settings
dataset_type = "StitchDataset"
nsweeps = 1
data_root = "data/stitch_dataset"

db_sampler = dict(
    type="GT-AUG",
    enable=False,
    db_info_path="data/stitch_dataset/dbinfos_train_1sweeps_withvelo.pkl",
    sample_groups=[
        dict(car=2),
        dict(truck=3),
        dict(heavy_truck=6),
        dict(bus=4),        
        dict(motorcycle=5),
        # dict(bicycle=5),
        dict(pedestrian=1),
    ],
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                car=5,
                heavy_truck=5,
                truck=5,
                bus=5,
                # bicycle=5,
                motorcycle=5,
                pedestrian=5
            )
        ),
        dict(filter_by_difficulty=[-1],),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
)
train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.3925, 0.3925],
    global_scale_noise=[0.95, 1.05],
    # db_sampler=db_sampler,
    db_sampler=None,
    class_names=class_names,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
)
### whole range / voxel size / 16
### x: 85.3 -> 86.4 -> 87.6
### y: 77.7 -> 78.0 -> ##79.2 78.3 0.9
### z: 4.4
voxel_generator = dict(
    # range = [208411.9, 534700.0, 75.4, 208497.3, 534778.0, 79.8],
    range = [-6.4, -16.5, -5.6, 80.0, 62.7, -1.3],
    voxel_size=[0.1, 0.1, 0.2],
    max_points_in_voxel=10,
    max_voxel_num=[90000, 120000],
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

train_anno = "data/stitch_dataset/infos_train_01sweeps_withvelo_filter_True.pkl"
val_anno = "data/stitch_dataset/infos_val_01sweeps_withvelo_filter_True.pkl"
test_anno = None

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    # workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)



optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.001, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
# yapf:enable
# runtime settings
total_epochs = 1
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None  
workflow = [('train', 1)]
