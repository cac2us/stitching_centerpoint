# modified from the single_inference.py by @muzi2045
from det3d.torchie.trainer import load_checkpoint
from det3d.models import build_detector
from det3d.utils.simplevis import nuscene_vis
from det3d.datasets.pipelines.preprocess import Voxelization
from det3d.torchie import Config
from tqdm import tqdm 
import numpy as np
import copy
import pickle 
import open3d as o3d
import cv2
import argparse
import torch
import time 
import os 
import json

STITCH_TRACKING_NAMES = [
    'car',
    'truck',
    'bus',
    'motorcycle',
    'pedestrian',
]
voxel_generator = None 
model = None 
device = None 

def initialize_model(args):
    global model, voxel_generator  
    cfg = Config.fromfile(args.config)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location="cpu")
    # print(model)
    if args.fp16:
        print("cast model to fp16")
        model = model.half()

    model = model.cuda()
    model.eval()

    global device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    range = cfg.voxel_generator.range
    voxel_size = cfg.voxel_generator.voxel_size
    max_points_in_voxel = cfg.voxel_generator.max_points_in_voxel
    max_voxel_num = cfg.voxel_generator.max_voxel_num
    cfg_dict = {
        'voxel_size':voxel_size,
        'range':range,
        'max_num_points':max_points_in_voxel,
        'max_voxel_num':max_voxel_num
    }
    voxel_generator = Voxelization(**{'cfg': cfg.voxel_generator})
    voxel_generator = voxel_generator.voxel_generator
    return model 

def voxelization(points, voxel_generator):
    voxel_output = voxel_generator.generate(points)  
    voxels, coords, num_points = \
        voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']

    return voxels, coords, num_points  

def _process_inputs(points, fp16):
    voxels, coords, num_points = voxel_generator.generate(points, max_voxels=120000)
    num_voxels = np.array([voxels.shape[0]], dtype=np.int32)
    grid_size = voxel_generator.grid_size
    coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values = 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    voxels = torch.tensor(voxels, dtype=torch.float32, device=device)
    coords = torch.tensor(coords, dtype=torch.int32, device=device)
    num_points = torch.tensor(num_points, dtype=torch.int32, device=device)
    num_voxels = torch.tensor(num_voxels, dtype=torch.int32, device=device)

    if fp16:
        voxels = voxels.half()

    inputs = dict(
            points = [torch.tensor(points).cuda()],
            voxels = voxels,
            num_points = num_points,
            num_voxels = num_voxels,
            coordinates = coords,
            shape = [grid_size]
        )

    return inputs 

def run_model(points, fp16=False):
    with torch.no_grad():
        data_dict = _process_inputs(points, fp16)
        outputs = model(data_dict, return_loss=False)[0]

    return {'boxes': outputs['box3d_lidar'].cpu().numpy(),
        'scores': outputs['scores'].cpu().numpy(),
        'classes': outputs['label_preds'].cpu().numpy()}

def process_example(points, fp16=False):
    output = run_model(points, fp16)

    return output

def load_points_stitch(pcd_path):
    # rotate 15
    pcd = o3d.io.read_point_cloud(pcd_path)
    T = np.eye(4)
    T[:3, :3] = pcd.get_rotation_matrix_from_xyz((np.pi / 13.5, 0, 0))
    T[0, 3] = 0
    T[1, 3] = 0
    T[2, 3] = 5.7
    pcd = copy.deepcopy(pcd).transform(T)
    points_pre = np.asarray(pcd.points)
    o3d.io.write_point_cloud('./temp.pcd', pcd)
    points_pre = np.fromfile('./temp.pcd', dtype=np.float32)
    # read file from stitch
    while len(points_pre) < 192000:
        points_pre = np.append(points_pre, np.array([0]))
        if len(points_pre) == 192000:
            break
    while len(points_pre) > 192000:
        points_pre = np.delete(points_pre, 0)
        if len(points_pre) == 192000:
            break
    points = points_pre.reshape(-1, 3)

    return points

def save_pred(pred):
    dets = [v for _, v in pred.items()]
    det_name = [i for i, _ in pred.items()]
    
    nusc_annos = {
            "results": {},
            "meta": None,
        }
    for k, det in enumerate(dets):
        annos = []
        boxes = det['boxes']
        name = det_name[k]
        for i, box in enumerate(boxes):
            nusc_anno = {
                    "sample_token": name,
                    "translation": box[:3].tolist(),
                    "size": box[3:6].tolist(),
                    "rotation": box[-1:].tolist(),
                    "velocity": box[6:8].tolist(),
                    "detection_name": STITCH_TRACKING_NAMES[det['classes'][i]],
                    "detection_score": float(det['scores'][i]),
                    "attribute_name": 'None',
            }
            annos.append(nusc_anno)
        nusc_annos["results"].update({name: annos})

    nusc_annos["meta"] = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

    # with open(os.path.join("prediction.pkl"), "wb") as f:
    #     pickle.dump(pred, f)

    with open('prediction.json', "w") as f:
            json.dump(nusc_annos, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CenterPoint")
    parser.add_argument("config", help="path to config file")
    parser.add_argument(
        "--checkpoint", help="the path to checkpoint which the model read from", default=None, type=str
    )
    parser.add_argument('--input_data_dir', type=str, required=True)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--threshold', default=0.5)
    parser.add_argument('--visual', action='store_true')
    parser.add_argument("--online", action='store_true')
    parser.add_argument('--num_frame', default=-1, type=int)
    args = parser.parse_args()

    print("Please prepare your point cloud in waymo format and save it as a pickle dict with points key into the {}".format(args.input_data_dir))
    print("One point cloud should be saved in one pickle file.")
    print("Download and save the pretrained model at {}".format(args.checkpoint))

    # Run any user-specified initialization code for their submission.
    model = initialize_model(args)

    latencies = []
    visual_dicts = []
    pred_dicts = {}
    counter = 0 
    predictions = {}
    os.makedirs('demo',exist_ok=True)
    for frame_name in tqdm(sorted(os.listdir(args.input_data_dir))):
        if counter == args.num_frame:
            break
        else:
            counter += 1 

        pc_name = os.path.join(args.input_data_dir, frame_name)

        points = load_points_stitch(pc_name)
        bev_map = nuscene_vis(points)

        detections = process_example(points, args.fp16)
        predictions.update({frame_name: detections})
        pred_boxes = detections['boxes'][:, [0, 1, 2, 3, 4, 5, -1]]
        bev_map = nuscene_vis(points, pred_boxes)
        cv2.imwrite('./demo/test_%02d.png' % counter, bev_map)
    save_pred(predictions)
