from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from det3d.torchie.trainer import load_checkpoint
from det3d.utils.simplevis import nuscene_vis
import torch 
from copy import deepcopy 
import numpy as np
# import open3d.ml.torch as ml3d
# import open3d as o3d

IDX = 0

@DETECTORS.register_module
class VoxelNet(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(VoxelNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        
    def extract_feat(self, data):
        if 'voxels' not in data:
            output = self.reader(data['points'])    
            voxels, coors, shape = output 

            data = dict(
                features=voxels,
                coors=coors,
                batch_size=len(data['points']),
                input_shape=shape,
                voxels=voxels
            )
            input_features = voxels
        else:
            data = dict(
                features=data['voxels'],
                num_voxels=data["num_points"],
                coors=data["coordinates"],
                batch_size=len(data['points']),
                input_shape=data["shape"][0],
            )
            input_features = self.reader(data["features"], data['num_voxels'])

        x, voxel_feature = self.backbone(
                input_features, data["coors"], data["batch_size"], data["input_shape"]
            )
        ### here
        if self.with_neck:
            x = self.neck(x)

        return x, voxel_feature

    def forward(self, example, return_loss=True, **kwargs):
        x, _ = self.extract_feat(example)
        preds, _ = self.bbox_head(x)
        if False:
            import cv2
            b_size = len(example['points'])
            for idx in range(b_size):
                points = example['points'][idx].detach().cpu().numpy()
                num_gt = len(example['gt_boxes_and_cls'][0].sum(1).nonzero())
                gt_boxes = example['gt_boxes_and_cls'][idx][:num_gt][:, :7].cpu().detach().numpy()
                bev_map = nuscene_vis(points, gt_boxes)
                cv2.imwrite('test_%02d.png' % idx, bev_map)
            import pdb; pdb.set_trace()
        
        # # convert tensor to numpy array
        # test_point = example['points'][0][:,:3]
        # test_point = test_point.detach().cpu().numpy()        
        # # create an Open3D point cloud from the numpy array
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(test_point)
        # o3d.io.write_point_cloud("points.pcd", pcd)
        # test_bb = example['gt_boxes_and_cls']
        # test_bb = test_bb.detach().cpu().numpy()
        # np.save("box", test_bb)

        if return_loss:
            return self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            abc = self.bbox_head.predict(example, preds, self.test_cfg)
            if True:
                global IDX
                import cv2
                b_size = len(example['points'])
                for idx in range(b_size):
                    points = example['points'][idx].detach().cpu().numpy()
                    gt_boxes = abc[idx]['box3d_lidar'][:, [0, 1, 2, 3, 4, 5, -1]].cpu().detach().numpy()
                    bev_map = nuscene_vis(points, gt_boxes)
                    cv2.imwrite('val/test_%02d.png' % IDX, bev_map)
                    IDX += 1
            return abc

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        x, voxel_feature = self.extract_feat(example)
        bev_feature = x 
        preds, final_feat = self.bbox_head(x)

        if return_loss:
            # manual deepcopy ...
            new_preds = []
            for pred in preds:
                new_pred = {} 
                for k, v in pred.items():
                    new_pred[k] = v.detach()
                new_preds.append(new_pred)

            boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

            return boxes, bev_feature, voxel_feature, final_feat, self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            boxes = self.bbox_head.predict(example, preds, self.test_cfg)
            return boxes, bev_feature, voxel_feature, final_feat, None 