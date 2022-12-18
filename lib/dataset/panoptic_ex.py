# Copyright 2021 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

from dataset.JointsDataset import JointsDataset
from utils.transforms import projectPoints


logger = logging.getLogger(__name__)
import time

# seq1: default dataset for debugging. (usually not use filter_valid_observations together)
# seq2: val dataset is suitable for valid filtering (usually use filter_valid_observations together)
# all: all datasets for the paper
# dbg: same train and val dataset as seq1 and seq2
TRAIN_LISTS = {
    'seq1': [
        '160906_pizza1',  # exchange with val
    ],
    'seq2' : [
        '160906_pizza1',  # exchange with val
    ],
    'all': [
        '160422_ultimatum1',
        '160224_haggling1',
        '160226_haggling1',
        '161202_haggling1',
        '160906_ian1',
        '160906_ian2',
        '160906_ian3',
        '160906_band1',
        '160906_band2',
        # '160906_band3',
    ],
    'dbg' : [
        '160906_pizza1',  # exchange with val
    ],
    'seq2-2' : [
        '160906_pizza1',
        '160906_ian2'
    ],
    'seq2-3' : [
        '160906_pizza1',
        '160906_ian2', 
        # '160224_haggling1',  # TODO: fix this dataset
        '160226_haggling1'
    ],
    'seq2-4' : [
        '160906_pizza1',
        '160906_ian2', 
        '160226_haggling1',
        '161202_haggling1',
    ],
    'seq2-5' : [
        '160906_pizza1',
        '160906_ian2', 
        '160226_haggling1',
        '161202_haggling1',
        '160422_ultimatum1',
    ],
    'seq2-6' : [
        '160906_pizza1',
        '160906_ian2', 
        '160226_haggling1',
        '161202_haggling1',
        '160422_ultimatum1',
        '160906_ian1',
    ],
    'seq2-7' : [
        '160906_pizza1',
        '160906_ian2', 
        '160226_haggling1',
        '161202_haggling1',
        '160422_ultimatum1',
        '160906_ian1',
        '160906_ian2',
    ],
    'seq2-8' : [
        '160906_pizza1',
        '160906_ian2', 
        '160226_haggling1',
        '161202_haggling1',
        '160422_ultimatum1',
        '160906_ian1',
        '160906_ian2',
        '160906_ian3',
    ],
    'ian-1': [
        '160906_ian1'
    ],
    'ian-2': [
        '160906_ian1',
        '160906_ian2',
    ],
    'ian-3': [
        '160906_ian1',
        '160906_ian2',
        '160906_ian3'
    ],
    
}
VAL_LISTS = {
    'seq1': ['160422_haggling1'],
    'seq2': ['160906_ian5'],
    'all': ['160906_pizza1', '160422_haggling1', '160906_ian5', '160906_band4'],
    'dbg' : [ '160906_pizza1' ],
    'seq2-2' : ['160906_ian5'],
    'seq2-3' : ['160906_ian5'],
    'seq2-4' : ['160906_ian5'],
    'seq2-5' : ['160906_ian5'],
    'seq2-6' : ['160906_ian5'],
    'seq2-7' : ['160906_ian5'],
    'seq2-8' : ['160906_ian5'],
    'ian-1' : ['160906_ian5'],
    'ian-2' : ['160906_ian5'],
    'ian-3' : ['160906_ian5'],
    'hag' : ['160422_haggling1'],
    'band' : ['160906_band4'],
    'all-val': [
        '160422_ultimatum1',
        '160224_haggling1',
        '160226_haggling1',
        '161202_haggling1',
        '160906_ian1',
        '160906_ian2',
        '160906_ian3',
        '160906_band1',
        '160906_band2',
        # '160906_band3',
    ],
}

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


class Panoptic(JointsDataset):
    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__(cfg, image_set, is_train, transform)
        self.pixel_std = 200.0
        self.joints_def = JOINTS_DEF
        self.limbs = LIMBS
        self.num_joints = len(JOINTS_DEF)
        self.MAX_DATA_NUM = cfg.DATASET.MAX_DATA_NUM

        if cfg.DATASET.SUBSET_SELECTION is None:
            dataset_selection = 'all'
        else:
            dataset_selection = cfg.DATASET.SUBSET_SELECTION

        self.filter_valid_observations = cfg.DATASET.FILTER_VALID_OBSERVATIONS

        if self.image_set == 'train' or self.image_set == 'save_voxel_pred_loaded':
            self.sequence_list = TRAIN_LISTS[dataset_selection]
            self._interval = 3
            self.cam_list = [(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)][
                            :self.num_views]
            self.num_views = len(self.cam_list)
        elif self.image_set == 'validation':
            self.sequence_list = VAL_LISTS[dataset_selection]
            self._interval = 12
            self.cam_list = [(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)][
                            :self.num_views]
            self.num_views = len(self.cam_list)

        self.db_file = 'group_{}_cam{}.pkl'.\
            format(self.image_set, self.num_views)
        self.db_file = os.path.join(self.dataset_root, self.db_file)

        close_full_dataset = False
        if not close_full_dataset \
            and osp.exists(self.db_file):
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
        self.db_size = len(self.db)

    def _get_db(self):
        time_start = time.time()
        temp_test_num = self.MAX_DATA_NUM
        print(' * MAX_DATA_NUM', temp_test_num)

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
                # one frame of different cameras
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
                    
                    if (temp_test_num is not None) and seq_count[seq] > temp_test_num:
                        break


                if (temp_test_num is not None) and seq_count[seq] > temp_test_num:
                    break
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

    # def loading_while(self, saving_path):
    #     try:
    #         temp = np.load(saving_path + '.npz')
    #         return temp
    #     except Exception as e:
    #         print('loading error, retrying loading')
    #         return None

    def __getitem__(self, idx):
        input, meta = [], []
        for k in range(self.num_views):
            i, m = super().__getitem__(self.num_views * idx + k)
            input.append(i)
            meta.append(m)
        return input, meta

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
            pred = pred[pred[:, 0, 3] >= 0]
            for pose in pred:
                mpjpes = []
                for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                    vis = gt_vis[:, 0] > 0
                    mpjpe = np.mean(np.sqrt(
                        np.sum((pose[vis, 0:3] - gt[vis]) ** 2, axis=-1)))
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

        mpjpe_threshold = np.arange(25, 155, 25)
        aps = []
        recs = []
        for t in mpjpe_threshold:
            ap, rec = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            recs.append(rec)

        return \
            aps, \
            recs, \
            self._eval_list_to_mpjpe(eval_list), \
            self._eval_list_to_recall(eval_list, total_gt)

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
