# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os.path as osp
import numpy as np
import json_tricks as json
import pickle
import logging
import os
import copy
import torch

from dataset.JointsDataset import JointsDataset
from utils.vis import save_batch_image_with_joints_multi,save_ref_points_with_gt
from utils.transforms import projectPoints
from prettytable import PrettyTable


logger = logging.getLogger(__name__)
import time

TRAIN_LIST = [
    '160422_ultimatum1',
    '160224_haggling1',
    '160226_haggling1',
    '161202_haggling1',
    '160906_ian1',
    '160906_ian2',
    '160906_ian3',
    '160906_band1',
    '160906_band2',
    '160906_band3',
]
TRAIN_SEQ = {
    'seq0': TRAIN_LIST,
    'seq1': [
        '160422_ultimatum1',
        '160224_haggling1',
        '160226_haggling1',
        '161202_haggling1',
        '160906_ian1',
        '160906_ian2',
        '160906_ian3',
        '160906_band1',
        '160906_band2',
    ]
}

VAL_LIST = ['160906_pizza1', '160422_haggling1', '160906_ian5', '160906_band4']

JOINTS_DEF = {
    'neck': 0,
    'nose': 1,
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
    # 'l-eye': 15,
    # 'l-ear': 16,
    # 'r-eye': 17,
    # 'r-ear': 18,
}

LIMBS = [[0, 1],
         [0, 2],
         [0, 3],
         [3, 4],
         [4, 5],
         [0, 9],
         [9, 10],
         [10, 11],
         [2, 6],
         [2, 12],
         [6, 7],
         [7, 8],
         [12, 13],
         [13, 14]]

CAM_LIST={
    'CMU0' : [(0, 3), (0, 6),(0, 12),(0, 13), (0, 23)], # 5
    'CMU0ex' : [(0, 3), (0, 6), (0, 12),(0, 13) (0, 23), (0, 10), (0, 16)], # 7 add 10 & 16
    'CMU1' : [(0, 1),(0, 2),(0, 3),(0, 4),(0, 6),(0, 7),(0, 10)],  #7
    'CMU2' : [(0, 12), (0, 16), (0, 18), (0, 19), (0, 22), (0, 23), (0, 30)], #7
    'CMU3' : [(0, 10), (0, 12), (0, 16), (0, 18)], #4
    'CMU4' : [(0, 6), (0, 7), (0, 10), (0, 12), (0, 16), (0, 18), (0, 19), (0, 22), (0, 23), (0, 30)], #10
}

class Panoptic(JointsDataset):
    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__(cfg, image_set, is_train, transform)
        self.pixel_std = 200.0
        self.joints_def = JOINTS_DEF
        self.limbs = LIMBS
        self.num_joints = len(JOINTS_DEF)
        self.save_result = not(cfg.DATASET.SAVE_RESULT is None)
        self.save_suffix = cfg.DATASET.SAVE_RESULT
        self.data_seq = cfg.DATASET.DATA_SEQ
        self.train_cam_seq = cfg.DATASET.TRAIN_CAM_SEQ
        self.test_cam_seq = cfg.DATASET.TEST_CAM_SEQ
        self.show_camera_detail = cfg.DATASET.CAMERA_DETAIL
        self.cam_seq = self.test_cam_seq if self.image_set == 'validation' else self.train_cam_seq
        if self.image_set == 'train':
            self.sequence_list = TRAIN_SEQ['seq1'] #! fix for only seq1
            self._interval = 3
            if self.num_views:
                self.cam_list = CAM_LIST[self.cam_seq][:self.num_views]
                self.num_views = len(self.cam_list)
            else:
                self.cam_list = CAM_LIST[self.cam_seq]
                self.num_views = len(self.cam_list)

        elif self.image_set == 'validation':
            self.sequence_list = VAL_LIST
            self._interval = 12
            if self.num_views:
                self.cam_list = CAM_LIST[self.cam_seq][:self.num_views]
                self.num_views = len(self.cam_list)
            else:
                self.cam_list = CAM_LIST[self.cam_seq]
                self.num_views = len(self.cam_list)

        self.db_file = 'group_{}_cam{}_{}.pkl'.\
            format(self.image_set, self.cam_seq, self.num_views)
        self.db_file = os.path.join(self.dataset_root, self.db_file)

        if osp.exists(self.db_file) and False:
            info = pickle.load(open(self.db_file, 'rb'))
            assert info['sequence_list'] == self.sequence_list
            assert info['interval'] == self._interval
            assert info['cam_list'] == self.cam_list
            self.db = info['db']
        else:
            self.db = self._get_db()
            info = {
                'sequence_list': self.sequence_list,
                'interval': self._interval,
                'cam_list': self.cam_list,
                'db': self.db
            }
            pickle.dump(info, open(self.db_file, 'wb'))
        # self.db = self._get_db()
        self.db_size = len(self.db)

    def _get_db(self):
        time_start = time.time()

        width = 1920
        height = 1080
        db = []
        seq_count = {}
        # all the sequence: different datasets
        for seq in self.sequence_list: 
            # for a specific dataset
            cameras = self._get_cam(seq)
            cam_num = len(cameras)
            curr_anno = osp.join(self.dataset_root,
                                 seq, 'hdPose3d_stage1_coco19')
            anno_files = sorted(glob.iglob('{:s}/*.json'.format(curr_anno)))

            seq_count[seq] = 0
            for i, file in enumerate(anno_files):
                if i % self._interval == 0:
                    with open(file) as dfile:
                        bodies = json.load(dfile)['bodies']
                    if len(bodies) == 0:
                        continue
                    
                    # check situation of different cameras
                    all_people_observable = []
                    for k, v in cameras.items():
                        postfix = osp.basename(file).replace('body3DScene', '')
                        prefix = '{:02d}_{:02d}'.format(k[0], k[1])
                        image = osp.join(seq, 'hdImgs', prefix,
                                         prefix + postfix)
                        image = image.replace('json', 'jpg')

                        all_poses_3d = []
                        all_poses_vis_3d = []
                        all_poses = []
                        all_poses_vis = []
                        for body in bodies:
                            pose3d = np.array(body['joints19'])\
                                .reshape((-1, 4))
                            pose3d = pose3d[:self.num_joints]

                            joints_vis = pose3d[:, -1] > 0.1

                            if not joints_vis[self.root_id]:
                                continue

                            # Coordinate transformation
                            M = np.array([[1.0, 0.0, 0.0],
                                          [0.0, 0.0, -1.0],
                                          [0.0, 1.0, 0.0]])
                            pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)

                            all_poses_3d.append(pose3d[:, 0:3] * 10.0)
                            all_poses_vis_3d.append(
                                np.repeat(
                                    np.reshape(
                                        joints_vis, (-1, 1)), 3, axis=1))

                            pose2d = np.zeros((pose3d.shape[0], 2))
                            pose2d[:, :2] = projectPoints(
                                pose3d[:, 0:3].transpose(), v['K'], v['R'],
                                v['t'], v['distCoef']).transpose()[:, :2]
                            x_check = \
                                np.bitwise_and(pose2d[:, 0] >= 0,
                                               pose2d[:, 0] <= width - 1)
                            y_check = \
                                np.bitwise_and(pose2d[:, 1] >= 0,
                                               pose2d[:, 1] <= height - 1)
                            check = np.bitwise_and(x_check, y_check)
                            joints_vis[np.logical_not(check)] = 0

                            all_poses.append(pose2d)
                            all_poses_vis.append(
                                np.repeat(
                                    np.reshape(
                                        joints_vis, (-1, 1)), 2, axis=1))

                        all_people_observable.append(all_poses_vis)
                        # check if there are any false
                        # for this camera, can all the bodies be visible?
                        # all_observed = True
                        # for arr in all_poses_vis:
                            # fail_pos = np.where(arr.reshape(-1)==False)
                            # if len(fail_pos) > 0:
                                # all_observed = False
                                # break

                        if len(all_poses_3d) > 0:
                            our_cam = {}
                            our_cam['R'] = v['R']
                            our_cam['T'] = -np.dot(
                                v['R'].T, v['t']) * 10.0  # cm to mm
                            our_cam['standard_T'] = v['t'] * 10.0
                            our_cam['fx'] = np.array(v['K'][0, 0])
                            our_cam['fy'] = np.array(v['K'][1, 1])
                            our_cam['cx'] = np.array(v['K'][0, 2])
                            our_cam['cy'] = np.array(v['K'][1, 2])
                            our_cam['k'] = v['distCoef'][[0, 1, 4]]\
                                .reshape(3, 1)
                            our_cam['p'] = v['distCoef'][[2, 3]]\
                                .reshape(2, 1)

                            db.append({
                                'key': "{}_{}{}".format(
                                    seq, prefix, postfix.split('.')[0]),
                                'image': osp.join(self.dataset_root, image),
                                'joints_3d': all_poses_3d,
                                'joints_3d_vis': all_poses_vis_3d,
                                'joints_2d': all_poses,
                                'joints_2d_vis': all_poses_vis,
                                'camera': our_cam
                            })
                    
                    # now we have all the cameras, all the peoples, all the joints, obsevable situation
                    valid = True
                    self.filter_valid_observations = False
                    if self.filter_valid_observations:
                        all_people_observable_arr = np.array(all_people_observable)
                        if all_people_observable_arr.shape[-1] > 0:
                            # For each joint, we want at least 3 observations?
                            xy_people_joints_obnum = np.sum(all_people_observable_arr.swapaxes(0,3), -1)
                            people_joints_obnum = xy_people_joints_obnum[0]
                            people_joints_valid = people_joints_obnum > 2
                            if False in people_joints_valid:
                                # not valid!
                                # remove the last 5
                                valid = False
                        else:
                            valid = False

                    if valid:
                        seq_count[seq]=seq_count[seq]+cam_num
                    else:
                        db = db[:-cam_num]
        print("Dataset result:", seq_count)
        time_end = time.time()
        print("Loading time:", time_end-time_start)

        return db

    def _get_cam(self, seq):
        cam_file = osp.join(self.dataset_root, seq,
                            'calibration_{:s}.json'.format(seq))
        with open(cam_file) as cfile:
            calib = json.load(cfile)

        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]])
        cameras = {}
        for cam in calib['cameras']:
            if (cam['panel'], cam['node']) in self.cam_list:
                sel_cam = {}
                sel_cam['K'] = np.array(cam['K'])
                sel_cam['distCoef'] = np.array(cam['distCoef'])
                sel_cam['R'] = np.array(cam['R']).dot(M)
                sel_cam['t'] = np.array(cam['t']).reshape((3, 1))
                cameras[(cam['panel'], cam['node'])] = sel_cam
        return cameras

    def __getitem__(self, idx):
        input, target, weight, target_3d, meta, input_heatmap = [], [], [], [], [], []

        # if self.image_set == 'train':
        #     # camera_num = np.random.choice([5], size=1)
        #     select_cam = np.random.choice(self.num_views, size=5, replace=False)
        # elif self.image_set == 'validation':
        #     select_cam = list(range(self.num_views))

        for k in range(self.num_views):
            i, t, w, t3, m, ih = super().__getitem__(self.num_views * idx + k)
            if i is None:
                continue
            input.append(i)
            target.append(t)
            weight.append(w)
            target_3d.append(t3)
            meta.append(m)
            input_heatmap.append(ih)
        return input, target, weight, target_3d, meta, input_heatmap

    def __len__(self):
        return self.db_size // self.num_views

    def evaluate(self, preds):
        eval_list = []
        gt_num = self.db_size // self.num_views
        assert len(preds) == gt_num, 'number mismatch'

        total_gt = 0
        for i in range(gt_num):
            index = self.num_views * i
            db_rec = copy.deepcopy(self.db[index])
            joints_3d = db_rec['joints_3d']
            joints_3d_vis = db_rec['joints_3d_vis']

            if len(joints_3d) == 0:
                continue

            pred = preds[i].copy()
            

            if self.save_result:
                self.db[index]['joints_3d_voxelpose_pred'] = pred
                
            pred = pred[pred[:, 0, 3] >= 0]

            for pose in pred:
                mpjpes = []
                for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                    vis = gt_vis[:, 0] > 0
                    mpjpe = np.mean(np.sqrt(np.sum((pose[vis, 0:3] - gt[vis]) ** 2, axis=-1)))
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                score = pose[0, 4]
                eval_list.append({
                    "mpjpe": float(min_mpjpe),
                    "score": float(score),
                    "gt_id": int(total_gt + min_gt)
                })

            total_gt += len(joints_3d)
        
        if self.save_result:
            self.db_file = 'group_{}_cam{}_{}_{}.pkl'.format(self.image_set, self.cam_seq ,self.num_views, self.save_suffix)
            self.db_file = os.path.join(self.dataset_root, self.db_file)
            info = {
                'sequence_list': self.sequence_list,
                'interval': self._interval,
                'cam_list': self.cam_list,
                'db': self.db
            }
            print(f"load save_voxel_pred to {self.db_file}")
            pickle.dump(info, open(self.db_file, 'wb'))
        
        
        def calc_ap(eval_list, total_gt):
            mpjpe_threshold = np.arange(25, 155, 25)
            aps = []
            recs = []
            for t in mpjpe_threshold:
                ap, rec = self._eval_list_to_ap(eval_list, total_gt, t)
                aps.append(ap)
                recs.append(rec)
            mpjpe = self._eval_list_to_mpjpe(eval_list)
            recall500 = self._eval_list_to_recall(eval_list, total_gt)
            return aps, recs, mpjpe, recall500
        
        
        self.vis_camera_details = ['10-4'] #!DEBUG         
        if self.show_camera_detail:
            from datetime import datetime
            gt_list = []
            pd_list = []
            ob_ths = range(0,100,10)
            now = datetime.now()
            now_str = now.strftime("%Y%m%d-%H%M%S")

            def obs_num(gt_id,ob_th):
                return int(gt_list[gt_id]['joints_2d_vis_num'][int(np.ceil(self.num_joints*ob_th/100))])

            def vis_camera_detail(pd,gt,prefix):
                os.makedirs(f'{prefix}')

                for index,g in enumerate(gt):
                    id = g['image_id']
                    id_sub = g['image_id_sub']
                    inputs, _, _, _, metas, _ = self.__getitem__(id)
                    for k,(input,meta) in enumerate(zip(inputs,metas)):
                        joints = meta['joints'][id_sub]
                        joints_vis = meta['joints_vis'][id_sub]
                        vis_num = np.sum(joints_vis)/2
                        save_batch_image_with_joints_multi(input[None,...],joints[None,None,...],joints_vis[None,None,...],[1],f'{prefix}/{index}_view{k}(vis_num:{vis_num:.0f}).jpg')


                for index,g in enumerate(gt):
                    id = g['image_id']
                    id_sub = g['image_id_sub']
                    inputs, _, _, _, metas, _ = self.__getitem__(id)
                    num_person = 1
                    joints_3d = g['joints_3d']
                    joints_3d_vis = g['joints_3d_vis']
                    meta = {
                        'num_person':[num_person],
                        'joints_3d':joints_3d[None,None,...],
                        'joints_3d_vis':joints_3d_vis[None,None,...]
                    }
                    preds = []
                    mpjpe = []
                    suffix = ''
                    for pred in pd:
                        if pred['gt_id']==g['gt_id']:
                            preds.append(torch.tensor(pred['joints_3d'][:,0:3]))
                            mpjpe.append(pred['mpjpe'])


                    if len(preds)==0:
                        preds.append(torch.zeros(15,3))

                    preds = torch.stack(preds)
                    preds = preds[None,...]
                    if len(mpjpe) > 0:
                        save_ref_points_with_gt(preds,meta,f'{prefix}/{index}_3d_{len(mpjpe)}pred_mpjpe{np.mean(mpjpe):.2f}.jpg')
                    else:
                        save_ref_points_with_gt(preds,meta,f'{prefix}/{index}_3d_no_pred.jpg')
                    
                    
            
            total = 0
            for i in range(gt_num):
                index = self.num_views * i
                db_rec = copy.deepcopy(self.db[index])
                joints_3d = db_rec['joints_3d']
                joints_3d_vis = db_rec['joints_3d_vis']
                image_file = [self.db[i]['image'] for i in range(index,index+self.num_views)] 
                joints_2d = [self.db[i]['joints_2d'] for i in range(index,index+self.num_views)]
                joints_2d_vis = [self.db[i]['joints_2d_vis'] for i in range(index,index+self.num_views)]
                joints_2d_vis_sum  = np.sum(joints_2d_vis,axis=0)[...,0]
                joints_2d_vis_sum = np.sort(joints_2d_vis_sum,axis=1)
                for sub_index, (gt, gt_vis,gt_2d_vis_num) in enumerate(zip(joints_3d, joints_3d_vis,joints_2d_vis_sum)):
                    gt_list.append({
                        'gt_id':total,
                        'image_file':image_file,
                        "image_id":i,
                        'image_id_sub':sub_index,
                        "joints_2d":[joint_2d[sub_index] for joint_2d in joints_2d],
                        "joints_2d_vis":[joint_2d_vis[sub_index] for joint_2d_vis in joints_2d_vis],
                        "joints_3d":gt,
                        "joints_3d_vis":gt_vis,
                        "joints_2d_vis_num":gt_2d_vis_num,
                    })
                    total = total + 1
                pred = preds[i].copy()
                pred = pred[pred[:, 0, 3] >= 0]
                for pose in pred:
                    pd_list.append(pose)

            tb = PrettyTable()
            mpjpe_threshold = np.arange(25, 155, 25)
            tb.field_names = \
                ["camera observation rate","num","pred_num","gt_num"] + \
                [f'AP{i}' for i in mpjpe_threshold] + \
                [f'Recall{i}' for i in mpjpe_threshold] + \
                ['Recall500','MPJPE']
            for ob_th in ob_ths:
                pd_list_sep = [[] for _ in range(0,self.num_views+1)]
                gt_list_sep = [[] for _ in range(0,self.num_views+1)]
                for gt_id,gt in enumerate(gt_list):
                    gt_list_sep[obs_num(gt_id,ob_th)].append(gt)
                for pd_id,pred in enumerate(eval_list):
                    pd_list_sep[obs_num(pred["gt_id"],ob_th)].append({
                        **pred,
                        "joints_3d":pd_list[pd_id]
                    })
                for i in range(1,self.num_views+1):
                    if len(gt_list_sep[i])==0:
                        continue
                    aps, recs, mpjpe, recall500 = calc_ap(pd_list_sep[i], len(gt_list_sep[i]))
                    tb.add_row( 
                        [f'{100-ob_th}%' ,i, len(pd_list_sep[i]),len(gt_list_sep[i])] + 
                        [f'{ap * 100:.2f}' for ap in aps] +
                        [f'{re * 100:.2f}' for re in recs] +
                        [f'{recall500 * 100:.2f}',f'{mpjpe:.2f}']
                    )
                    if f'{100-ob_th}-{i}' in self.vis_camera_details:
                        prefix = f'./vis_results/{now_str}/camera_detail/{100-ob_th}-{i}/'
                        vis_camera_detail(pd_list_sep[i],gt_list_sep[i],prefix)

            aps, recs, mpjpe, recall500 = calc_ap(eval_list, total_gt)
            tb.add_row(
                ['all' ,'all' ,len(eval_list),total_gt] + 
                [f'{ap * 100:.2f}' for ap in aps] +
                [f'{re * 100:.2f}' for re in recs] +
                [f'{recall500 * 100:.2f}',f'{mpjpe:.2f}']
            )
            logger.info(tb)
        aps, recs, mpjpe, recall500 = calc_ap(eval_list, total_gt) 
        return aps, recs, mpjpe, recall500

    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt, threshold):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        total_num = len(eval_list)

        tp = np.zeros(total_num)
        fp = np.zeros(total_num)
        gt_det = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                tp[i] = 1
                gt_det.append(item["gt_id"])
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / (total_gt + 1e-5)
        precise = tp / (tp + fp + 1e-5)
        for n in range(total_num - 2, -1, -1):
            precise[n] = max(precise[n], precise[n + 1])

        precise = np.concatenate(([0], precise, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

        return ap, recall[-2]

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold=500):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        gt_det = []

        mpjpes = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                mpjpes.append(item["mpjpe"])
                gt_det.append(item["gt_id"])

        return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

    @staticmethod
    def _eval_list_to_recall(eval_list, total_gt, threshold=500):
        gt_ids = [e["gt_id"] for e in eval_list if e["mpjpe"] < threshold]

        return len(np.unique(gt_ids)) / total_gt




