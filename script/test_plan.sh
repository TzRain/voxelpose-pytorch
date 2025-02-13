#! /bin/bash -x

python test/evaluate.py --cfg configs/panoptic/resnet50/std64.yaml GPUS='0,1' DATASET.TRAIN_CAM_SEQ='CMU1' DATASET.TEST_CAM_SEQ='CMU1' DATASET.CAMERA_NUM=7 DATASET.SAVE_RESULT='voxel_64_pred_12-26' DEBUG.WANDB_NAME='std64 CMU1'
python test/evaluate.py --cfg configs/panoptic/resnet50/std64.yaml GPUS='0,1' DATASET.TRAIN_CAM_SEQ='CMU2' DATASET.TEST_CAM_SEQ='CMU2' DATASET.CAMERA_NUM=7 DATASET.SAVE_RESULT='voxel_64_pred_12-26' DEBUG.WANDB_NAME='std64 CMU2'
python test/evaluate.py --cfg configs/panoptic/resnet50/std64.yaml GPUS='0,1' DATASET.TRAIN_CAM_SEQ='CMU3' DATASET.TEST_CAM_SEQ='CMU3' DATASET.CAMERA_NUM=4 DATASET.SAVE_RESULT='voxel_64_pred_12-26' DEBUG.WANDB_NAME='std64 CMU3'
python test/evaluate.py --cfg configs/panoptic/resnet50/std64.yaml GPUS='0,1' DATASET.TRAIN_CAM_SEQ='CMU4' DATASET.TEST_CAM_SEQ='CMU4' DATASET.CAMERA_NUM=10 DATASET.SAVE_RESULT='voxel_64_pred_12-26' DEBUG.WANDB_NAME='std64 CMU4'
python test/evaluate.py --cfg configs/panoptic/resnet50/std64.yaml GPUS='0,1' DATASET.TRAIN_CAM_SEQ='CMU0' DATASET.TEST_CAM_SEQ='CMU0' DATASET.CAMERA_NUM=5 DATASET.SAVE_RESULT='voxel_64_pred_12-26' DEBUG.WANDB_NAME='std64 CMU0'