# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import argparse
import os
from tqdm import tqdm
from prettytable import PrettyTable
import copy

import _init_paths
from datetime import datetime
from core.config import config,update_config, update_config_dynamic_input
from utils.utils import create_logger, load_backbone_panoptic
import dataset
import pprint
import models
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    update_config_dynamic_input(rest)
    return args


def main():
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'eval_map')
    cfg_name = os.path.basename(args.cfg).split('.')[0]
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))
    if config.DEBUG.WANDB_KEY:
        wandb.login(key=config.DEBUG.WANDB_KEY)
    if config.DEBUG.WANDB_NAME:
        wandb.init(project="vp-val",name=config.DEBUG.WANDB_NAME)
    else:
        wandb.init(project="vp-val")

    gpus = [int(i) for i in config.GPUS.split(',')]
    logger.info('=> Loading data ..')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    logger.info('=> Constructing models ..')
    model = eval('models.' + config.MODEL + '.get_multi_person_pose_net')(
        config, is_train=True)
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    test_model_file = os.path.join(final_output_dir, config.TEST.MODEL_FILE)
    if config.TEST.MODEL_FILE and os.path.isfile(test_model_file):
        logger.info('=> load models state {}'.format(test_model_file))
        model.module.load_state_dict(torch.load(test_model_file),False)
    else:
        raise ValueError(f'Check the model file for testing! {test_model_file}')

    model.eval()
    preds = []
    now = datetime.now()
    now_str = now.strftime("%Y%m%d-%H%M%S")
    with torch.no_grad():
        pred_path = os.path.join(final_output_dir, f"{config.TEST.PRED_FILE}.npy")

        if config.TEST.PRED_FILE and os.path.isfile(pred_path):
            preds = np.load(pred_path)
            logger.info(f"=> load pred_file from {pred_path}")
        
        else:

            for i, (inputs, targets_2d, weights_2d, targets_3d, meta, input_heatmap) in enumerate(tqdm(test_loader)):
                if 'panoptic' in config.DATASET.TEST_DATASET:
                    pred, _, _, _, _, _ = model(views=inputs, meta=meta)
                elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:
                    pred, _, _, _, _, _ = model(meta=meta, input_heatmaps=input_heatmap)

                pred = pred.detach().cpu().numpy()
                for b in range(pred.shape[0]):
                    preds.append(pred[b])
            
            np.save(final_output_dir+'/{}.npy'.format(now_str),preds)
            logger.info(f"=> save pred_file with TEST.PRED_FILE={now_str}")

        tb = PrettyTable()
        if 'panoptic' in config.DATASET.TEST_DATASET:
            mpjpe_threshold = np.arange(25, 155, 25)
            aps, recs, mpjpe, recall500 = test_dataset.evaluate(preds)
            tb.field_names = \
                ["config_name"] + \
                [f'AP{i}' for i in mpjpe_threshold] + \
                [f'Recall{i}' for i in mpjpe_threshold] + \
                ['Recall500','MPJPE']
            tb.add_row( 
                [config.DEBUG.WANDB_NAME] + 
                [f'{ap * 100:.2f}' for ap in aps] +
                [f'{re * 100:.2f}' for re in recs] +
                [f'{recall500 * 100:.2f}',f'{mpjpe:.2f}']
            )
            logger.info(tb)
        else:
            actor_pcp, avg_pcp, bone_person_pcp, _ = test_dataset.evaluate(preds)
            tb.field_names = ['Bone Group'] + [f'Actor {i+1}' for i in range(len(actor_pcp))] + ['Average']
            for k, v in bone_person_pcp.items():
                tb.add_row([k] + [f'{i*100:.1f}' for i in v] + [f'{np.mean(v)*100:.1f}'])
            tb.add_row(['Total'] + [f'{i*100:.1f}' for i in actor_pcp] + [f'{avg_pcp*100:.1f}'])
            logger.info(tb)


if __name__ == "__main__":
    main()
