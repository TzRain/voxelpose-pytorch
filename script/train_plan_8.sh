#! /bin/bash -x

python run/train_3d.py --cfg configs/panoptic/resnet50/std64.yaml GPUS='0,1,2,3,4,5,6,7' DATASET.TRAIN_CAM_SEQ='CMU1' DATASET.TEST_CAM_SEQ='CMU1' DATASET.CAMERA_NUM=0  DEBUG.WANDB_NAME='std64(CM1)'
python run/train_3d.py --cfg configs/panoptic/resnet50/std64.yaml GPUS='0,1,2,3,4,5,6,7' DATASET.TRAIN_CAM_SEQ='CMU2' DATASET.TEST_CAM_SEQ='CMU2' DATASET.CAMERA_NUM=0  DEBUG.WANDB_NAME='std64(CM2)'
python run/train_3d.py --cfg configs/panoptic/resnet50/std64.yaml GPUS='0,1,2,3,4,5,6,7' DATASET.TRAIN_CAM_SEQ='CMU3' DATASET.TEST_CAM_SEQ='CMU3' DATASET.CAMERA_NUM=0  DEBUG.WANDB_NAME='std64(CM3)'
python run/train_3d.py --cfg configs/panoptic/resnet50/std64.yaml GPUS='0,1,2,3,4,5,6,7' DATASET.TRAIN_CAM_SEQ='CMU4' DATASET.TEST_CAM_SEQ='CMU4' DATASET.CAMERA_NUM=0  DEBUG.WANDB_NAME='std64(CM4)'


python run/train_3d.py --cfg configs/panoptic/resnet50/std64.yaml GPUS='0,1' DATASET.TRAIN_CAM_SEQ='CMU4' DATASET.TEST_CAM_SEQ='CMU4' DATASET.CAMERA_NUM=0  DEBUG.WANDB_NAME='std64(CM4)'
