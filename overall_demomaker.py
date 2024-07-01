#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import pickle as pkl

import numpy as np
import cv2

from open3d_vis import Visualizer, get_sample_data
from nuscenes import NuScenes
from simplevis import nuscene_vis
from scipy.interpolate import interp1d

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
import numpy as np # Scientific computing library for Python
 
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

def parse_func():
    cnt = 0
    track_json = json.load(open('track/tracking_result.json', 'r'))
    pred_pkl = pkl.load(open('stitching_prediction/final_track_results/pred_to_track.pickle', 'rb'))
    os.makedirs("gt_png",exist_ok=True)
    os.makedirs("over_ori",exist_ok=True)
    os.makedirs("pred_png",exist_ok=True)
    scene_token = list(pred_pkl.keys())[0]
    num_samp = len(os.listdir("stitching_dir"))
    sample_lists = track_json['results']
    sample_lists = list(sample_lists.keys())

    for s_idx in range(num_samp):
        samp_token = sample_lists[s_idx]
        track = track_json['results'][samp_token]
        lidar_dir = os.path.join('stitching_dir', samp_token)
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
        cv2.imwrite('./pred_png/%06d.png' % cnt, bev_img)

        # prediction demo
        trans_list, size_list, rot_list, id_list, cls_list, pred_list = \
            [], [], [], [], [], []
        print(cnt)
        for o_idx in range(len(track)):
            if track[o_idx]['tracking_score'] < 0.1:
                continue
            track_id = int(track[o_idx]['tracking_id'])
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
        if len(pred_list):
            print('pred available')

        preds_list = np.array(pred_list)
        if len(preds_list) and False:
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

        # strcnt = len(os.listdir("over_ori"))
        strcnt = str(cnt).zfill(6)
        cv2.imwrite(f'./over_ori/{strcnt}.png', bev_img)
        cnt += 1
        
def gen_video():
    import cv2
    import os

    image_folder = 'over_ori'
    video_name = 'video.avi'

    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 10, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    parse_func()
    gen_video()
