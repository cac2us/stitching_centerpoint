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
import importlib
import sys
from open3d_vis import Visualizer, get_sample_data
from nuscenes import NuScenes
from simplevis import nuscene_vis
from scipy.interpolate import interp1d
from stitching_prediction.data_st import *
from stitching_prediction.stiching_model import *
from stitching_prediction.utils import *
from stitching_prediction.stitching_prediction_2 import *

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck'
]

# 99.9 percentile of the l2 velocity error distribution (per clss / 0.5 second)
# This is an earlier statistcs and I didn't spend much time tuning it.
# Tune this for your model should provide some considerable AMOTA improvement
NUSCENE_CLS_VELOCITY_ERROR = {
  'car':4,
  'truck':4,
  'bus':5.5,
  'trailer':3,
  'pedestrian':1,
  'motorcycle':13,
  'bicycle':3,  
}

STITCH_TRACKING_NAMES = [
    'car',
    'car',
    'car',
    'car',
    'pedestrian',
    #  'car',
    #  'truck',
    #  'bus',
    #  'motorcycle',
    #  'pedestrian',
]
voxel_generator = None 
model = None 
device = None
vis_cnt = 0

class PubTracker(object):
  def __init__(self,  hungarian=False, max_age=0):
    self.hungarian = hungarian
    self.max_age = max_age

    self.NUSCENE_CLS_VELOCITY_ERROR = NUSCENE_CLS_VELOCITY_ERROR

    self.reset()
  
  def reset(self):
    self.id_count = 0
    self.tracks = []

  def step_centertrack(self, results, time_lag):
    if len(results) == 0:
      self.tracks = []
      return []
    else:
      temp = []
      for det in results:
        # filter out classes not evaluated for tracking 
        if det['detection_name'] not in NUSCENES_TRACKING_NAMES:
          continue 

        det['ct'] = np.array(det['translation'][:2])
        det['tracking'] = np.array(det['velocity'][:2]) * -1 * time_lag
        det['label_preds'] = NUSCENES_TRACKING_NAMES.index(det['detection_name'])
        temp.append(det)

      results = temp

    N = len(results)
    M = len(self.tracks)

    # N X 2 
    if 'tracking' in results[0]:
      dets = np.array(
      [ det['ct'] + det['tracking'].astype(np.float32)
       for det in results], np.float32)
    else:
      dets = np.array(
        [det['ct'] for det in results], np.float32) 

    item_cat = np.array([item['label_preds'] for item in results], np.int32) # N
    track_cat = np.array([track['label_preds'] for track in self.tracks], np.int32) # M

    max_diff = np.array([self.NUSCENE_CLS_VELOCITY_ERROR[box['detection_name']] for box in results], np.float32)

    tracks = np.array(
      [pre_det['ct'] for pre_det in self.tracks], np.float32) # M x 2

    if len(tracks) > 0:  # NOT FIRST FRAME
      dist = (((tracks.reshape(1, -1, 2) - \
                dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M
      dist = np.sqrt(dist) # absolute distance in meter

      invalid = ((dist > max_diff.reshape(N, 1)) + \
      (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0

      dist = dist  + invalid * 1e18
      if self.hungarian:
        dist[dist > 1e18] = 1e18
        matched_indices = linear_assignment(copy.deepcopy(dist))
      else:
        matched_indices = greedy_assignment(copy.deepcopy(dist))
    else:  # first few frame
      assert M == 0
      matched_indices = np.array([], np.int32).reshape(-1, 2)

    unmatched_dets = [d for d in range(dets.shape[0]) \
      if not (d in matched_indices[:, 0])]

    unmatched_tracks = [d for d in range(tracks.shape[0]) \
      if not (d in matched_indices[:, 1])]
    
    if self.hungarian:
      matches = []
      for m in matched_indices:
        if dist[m[0], m[1]] > 1e16:
          unmatched_dets.append(m[0])
        else:
          matches.append(m)
      matches = np.array(matches).reshape(-1, 2)
    else:
      matches = matched_indices

    ret = []
    for m in matches:
      # import pdb; pdb.set_trace()
      # print('m matches')
      track = results[m[0]]
      # print(track['sample_token'])
      track['tracking_id'] = self.tracks[m[1]]['tracking_id']      
      track['age'] = 1
      track['active'] = self.tracks[m[1]]['active'] + 1
      ret.append(track)

    for i in unmatched_dets:
      # import pdb; pdb.set_trace()
      # print("unmatched dets")
      track = results[i]
      # print(track['sample_token'])
      self.id_count += 1
      track['tracking_id'] = self.id_count
      track['age'] = 1
      track['active'] =  1
      ret.append(track)

    # still store unmatched tracks if its age doesn't exceed max_age, however, we shouldn't output 
    # the object in current frame 
    for i in unmatched_tracks:
      # import pdb; pdb.set_trace()
      # print("unmatched tracks")
      track = self.tracks[i]
      # print(track['sample_token'])
      if track['age'] < self.max_age:
        track['age'] += 1
        track['active'] = 0
        ct = track['ct']

        # movement in the last second
        if 'tracking' in track:
            offset = track['tracking'] * -1 # move forward 
            track['ct'] = ct + offset 
        ret.append(track)

    self.tracks = ret
    # print(ret)
    return ret

def greedy_assignment(dist):
  matched_indices = []
  if dist.shape[1] == 0:
    return np.array(matched_indices, np.int32).reshape(-1, 2)
  for i in range(dist.shape[0]):
    j = dist[i].argmin()
    if dist[i][j] < 1e16:
      dist[:, j] = 1e18
      matched_indices.append([i, j])
  return np.array(matched_indices, np.int32).reshape(-1, 2)


def load_pts(pts_filename):
    if pts_filename.endswith('.npy'):
        points_pre = np.load(pts_filename)
    else:
        points_pre = np.fromfile(pts_filename, dtype=np.float32)
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

def boxes_to_array(boxes):
    locs = np.array([b.center for b in boxes]).reshape(-1, 3)
    dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
    rots = np.array([b.orientation.yaw_pitch_roll[0]
                     for b in boxes]).reshape(-1, 1)
    gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
    return gt_boxes
 
def get_quaternion_from_euler(yaw, roll=0, pitch=0):
    """
    Convert an Euler angle to a quaternion.

    Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]

def visualize():
    global vis_cnt
    track_json = json.load(open('track/tracking_result.json', 'r'))
    pred_pkl = pickle.load(open('final_track_results/pred_to_track.pickle', 'rb'))
    # import pdb; pdb.set_trace()
    os.makedirs("over_ori",exist_ok=True)
    os.makedirs("pred_png",exist_ok=True)
    scene_token = list(pred_pkl.keys())[0]
    sample_lists = track_json['results']
    sample_lists = list(sample_lists.keys())
    num_samp = len(track_json)
    samp_token = sample_lists[-1]
    track = track_json['results'][samp_token]
    lidar_dir = os.path.join(args.input_data_dir, samp_token)
    points = load_pts(lidar_dir)
    # tracking demo
    trans_list, size_list, rot_list, real_rot_list, id_list_tr, cls_list = \
        [], [], [], [], [], []
    for o_idx in range(len(track)):
        if track[o_idx]['tracking_score'] < 0.2:
            continue
        trans_list.append(track[o_idx]['translation'])
        size_list.append(track[o_idx]['size'])
        yaw_angle = track[o_idx]['rotation']
        quat_yaw_angle = get_quaternion_from_euler(yaw_angle[0])
        rot_list.append(quat_yaw_angle)
        real_rot_list.append(yaw_angle[0])
        id_list_tr.append(int(track[o_idx]['tracking_id']))
        cls_list.append(track[o_idx]['tracking_name'])

    pred = (trans_list, size_list, rot_list, id_list_tr, cls_list)
    
    if len(pred[0]) != 0:
        gt_boxes_tr = []
        for index in range(len(real_rot_list)):
            gt_boxes_tr.append(pred[0][index]+pred[1][index]+[real_rot_list[index]])
        gt_boxes_tr = np.array(gt_boxes_tr)[:,[0,1,2,3,4,5,6]]
    bev_img = nuscene_vis(points, gt_boxes_tr)
    cv2.imwrite('./pred_png/%06d.png' % vis_cnt, bev_img)

    # prediction demo
    trans_list, size_list, rot_list, id_list, cls_list, pred_list = \
        [], [], [], [], [], []
    for o_idx in range(len(track)):
        if track[o_idx]['tracking_score'] < 0.1:
            continue
        track_id = track[o_idx]['tracking_id']
        if track_id not in pred_pkl[scene_token]:
            continue
        if len(pred_pkl[scene_token][track_id][samp_token]) == 0:
            continue
        pred_arr = pred_pkl[scene_token][track_id][samp_token]
        pred_list.append(pred_arr['trajectory_pred'])

        trans_list.append(track[o_idx]['translation'])
        size_list.append(track[o_idx]['size'])
        yaw_angle = track[o_idx]['rotation']
        yaw_angle = [0]
        yaw_angle = get_quaternion_from_euler(yaw_angle[0])
        rot_list.append(yaw_angle)
        id_list.append(track_id)
        cls_list.append(track[o_idx]['tracking_name'])
    pred = (trans_list, size_list, rot_list, id_list, cls_list)
    # convert boxes, prediction globacl coor -> local coor


    preds_list = np.array(pred_list)
    if len(preds_list):
        new_preds_list = []
        for idx in range(len(preds_list)):
            xnew = np.linspace(preds_list[idx][:, 0].min(), preds_list[idx][:, 0].max(), 20)
            coefficients = np.polyfit(preds_list[idx][:, 0], preds_list[idx][:, 1], 2)
            func = np.poly1d(coefficients)
            ynew = func(xnew) - (func(preds_list[idx][0][0]) - preds_list[idx][0][1])
            new_pred = np.hstack([xnew.reshape(-1, 1), ynew.reshape(-1, 1)])
            new_preds_list.append(new_pred)
        preds_list = np.array(new_preds_list)
    bev_img = nuscene_vis(points, gt_boxes_tr, preds=preds_list, ids=id_list,
                            boxes_tr=gt_boxes_tr, ids_tr=id_list_tr)
    path = './over_ori/' + scene_token

    cv2.imwrite('./over_ori/%06d.png' % vis_cnt, bev_img)
    vis_cnt += 1

def initialize_model(args):
    global model, voxel_generator  
    cfg = Config.fromfile(args.config)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location="cpu")
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

def save_pred(frame_name, pred):
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

    return nusc_annos['results'][frame_name]

def order_check(input_data_list):
    if os.path.isfile(os.path.join('track', 'tracking_result.json')):
        with open(os.path.join('track', 'tracking_result.json'), "rb") as ff:
            tracks = json.load(ff)
        past_json = list(tracks['results'].keys())
        past_json_final = int(past_json[-1].split('.')[0])
        present_json_first = int(input_data_list[0].split('.')[0])
        import pdb; pdb.set_trace()
        if (past_json_final+1) == present_json_first:
          past_json.append(input_data_list[0])
        else:
          print("Please load continuous data")
          import pdb; pdb.set_trace()
    else:
        print("New data")


    return past_json

def track(frame_id, pred):
    scene_token = "iechhs0fjtjw7ttioqi5skr7ipuuekqv"
    tracker = PubTracker(max_age=5, hungarian=False)
    time_lag = 1
    if os.path.isfile(os.path.join('track', 'tracks.pickle')):
        with open(os.path.join('track', 'tracks.pickle'), "rb") as ff:
            tracks = pickle.load(ff)
        tracker.tracks = tracks['outputs']
        tracker.id_count = tracks['id_count']

    outputs = tracker.step_centertrack(pred, time_lag)
    annos = []
    for item in outputs:
        if item['active'] == 0:
            continue 
        nusc_anno = {
            "sample_token": frame_id,
            "translation": item["translation"],
            "size": item["size"],
            "rotation": item["rotation"],
            "tracking_id": str(item["tracking_id"]),
            "tracking_score": item["detection_score"],
            "tracking_name": item["detection_name"],
            "gt_box_loc": item["translation"],
            "gt_box_size": item["size"],
            "gt_box_rot": item["rotation"],
            "gt_class": item["detection_name"],
            "timestamp": int(frame_id.split(".")[0]),
            "scene_token": scene_token,
            "velocity": item["velocity"]
        }
        annos.append(nusc_anno)
    output_dict = {"id_count":tracker.id_count, "outputs":outputs}
    with open(os.path.join('track', 'tracks.pickle'), "wb") as f:
        pickle.dump(output_dict, f)
    return annos
    
def save_tracks(tracklets):
    with open(os.path.join('track', 'tracking_result.json'), "w") as f:
        json.dump(tracklets, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CenterPoint")
    parser.add_argument("config", help="path to config file")
    parser.add_argument(
        "--checkpoint", help="the path to checkpoint which the model read from", default='./stitching_centerpoint_test.pth', type=str
    )
    parser.add_argument('--input_data_dir', type=str, default='./stitching_dir/')
    parser.add_argument('--start_frame', type=str, default=0)
    parser.add_argument('--end_frame', type=str, default=10)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--threshold', default=0.5)
    parser.add_argument("--online", action='store_true')

    #For prediction
    parser.add_argument(
        "-j", "--json_file", default="Stitch_gt_forpred.json", type=str, metavar="JSONFILE", help="json file path"
    )
    parser.add_argument(
        "-sp", "--split", default="gt_val", type=str, metavar="SPLIT", help="data split type"
    )
    parser.add_argument(
        "-p", "--preprocess_path", default="preprocess", type=str, metavar="PREPROCESS_PATH", help="preprocess path"
    )
    parser.add_argument(
        "-m", "--model", default="lanegcn", type=str, metavar="MODEL", help="model name"
    )
    parser.add_argument("--eval", '-e', action="store_true")
    parser.add_argument("--valeval", '-ve', default=True, action="store_true")
    parser.add_argument("--traineval", '-te', action="store_true")
    parser.add_argument("--test", '-t', action="store_true")
    parser.add_argument(
        "--ckpt", '-c', default=30, type=int, metavar="CKPT_NUM", help="checkpoint number"
    )
    parser.add_argument(
        "--resume", default="", type=str, metavar="RESUME", help="checkpoint path"
    )
    parser.add_argument(
        "--weight", default="", type=str, metavar="WEIGHT", help="checkpoint path"
    )
    parser.add_argument(
        "--re_epoch", '-re', default="", type=str, metavar="WEIGHT", help="checkpoint path"
    )
    parser.add_argument(
        "-f", "--pred_to_track_file", default="pred_to_track", type=str, metavar="PRED2TRACK_FILE", help="pred2track file name"
    )
    parser.add_argument(
        "-fp", "--final_track_file_path", default="final_track_results", type=str, metavar="TRACK_FILE_PATH", help="pred2track file path"
    )

    args = parser.parse_args()

    print("Please prepare your point cloud and save it as a pickle dict with points key into the {}".format(args.input_data_dir))
    print("One point cloud should be saved in one pickle file.")
    print("Download and save the pretrained model at {}".format(args.checkpoint))

    # Run any user-specified initialization code for their submission.
    model = initialize_model(args)
    os.makedirs("track",exist_ok=True)

    latencies = []
    visual_dicts = []
    pred_dicts = {}
    counter = 0 
    predictions = {}
    past_frames = []
    tracklets = {
        "results": {},
        "meta": None,
    }

    tracking_results_list = []
    start_frame = int(args.start_frame)
    end_frame = int(args.end_frame)
    input_data_list = []

    ## Select Frames
    for selected_frame in sorted(os.listdir(args.input_data_dir)):
      frame_num = int(selected_frame.split('.')[0])
      if (frame_num >= start_frame) and (frame_num <= end_frame):
        input_data_list.append(selected_frame)

    # order_check(input_data_list)
    input_data_list = sorted(input_data_list)
    for frame_name in tqdm(input_data_list):        
        pc_name = os.path.join(args.input_data_dir, frame_name)

        points = load_points_stitch(pc_name)
        bev_map = nuscene_vis(points)

        detections = process_example(points, args.fp16)
        predictions.update({frame_name: detections})
        # import pdb; pdb.set_trace()
        if len(predictions) == 6 :
            past_frames = list(predictions.keys())
            old_frame = past_frames[0]
            predictions.pop(old_frame)
        det = save_pred(frame_name, predictions)

        ## create tracks.pickle 
        tracks = track(frame_name, det)
        tracklets['results'].update({frame_name: tracks})
        if len(tracklets['results']) == 6 :
            past_tracks = list(tracklets['results'].keys())
            old_tracks = past_tracks[0]
            tracklets['results'].pop(old_tracks)

        ## create tracking_result.json
        save_tracks(tracklets)

        tracking_results_list.append(copy.deepcopy(tracklets))
        if len(tracking_results_list) < 5:
            continue

        csv_dict = Track2ArgoSteverGt(args=args, tracking_results_list=tracking_results_list)
        npz_dict = PrerenderStver(args=args, csv_dict=csv_dict)
        stitching_model_val = ValSt(args=args, npz_dict=npz_dict)
        Pred2TrackStver(args=args, stitching_model_val=stitching_model_val, tracking_results_list=tracking_results_list)

        del tracking_results_list[0]
        if args.visualize:
            visualize()
