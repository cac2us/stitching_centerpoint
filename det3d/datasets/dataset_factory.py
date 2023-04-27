from .nuscenes import NuScenesDataset
from .waymo import WaymoDataset
from .stitch import StitchDataset

dataset_factory = {
    "NUSC": NuScenesDataset,
    "WAYMO": WaymoDataset,
    "ST": StitchDataset
}


def get_dataset(dataset_name):
    return dataset_factory[dataset_name]
