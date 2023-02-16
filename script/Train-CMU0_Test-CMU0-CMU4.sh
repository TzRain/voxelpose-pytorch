#! /bin/bash -x
# python test/evaluate.py --cfg configs/panoptic/resnet50/std64.yaml GPUS='0,1,2,3' DATASET.TRAIN_CAM_SEQ='CMU0ex' DATASET.TEST_CAM_SEQ='CMU0ex' DATASET.CAMERA_NUM=3 DEBUG.WANDB_NAME='Train(CMU4) Test(CMU0ex3)'

# python test/evaluate.py --cfg configs/panoptic/resnet50/std64.yaml GPUS='0,1,2,3' DATASET.TRAIN_CAM_SEQ='CMU0ex' DATASET.TEST_CAM_SEQ='CMU0ex' DATASET.CAMERA_NUM=4 DEBUG.WANDB_NAME='Train(CMU4) Test(CMU0ex4)'

python test/evaluate.py --cfg configs/panoptic/resnet50/std64.yaml GPUS='0,1,2,3' DATASET.TRAIN_CAM_SEQ='CMU0ex' DATASET.TEST_CAM_SEQ='CMU0ex' DATASET.CAMERA_NUM=5 DEBUG.WANDB_NAME='Train(CMU0) Test(CMU0ex5)'

python test/evaluate.py --cfg configs/panoptic/resnet50/std64.yaml GPUS='0,1,2,3' DATASET.TRAIN_CAM_SEQ='CMU0ex' DATASET.TEST_CAM_SEQ='CMU0ex' DATASET.CAMERA_NUM=6 DEBUG.WANDB_NAME='Train(CMU0) Test(CMU0ex6)'

python test/evaluate.py --cfg configs/panoptic/resnet50/std64.yaml GPUS='0,1,2,3' DATASET.TRAIN_CAM_SEQ='CMU0ex' DATASET.TEST_CAM_SEQ='CMU0ex' DATASET.CAMERA_NUM=7 DEBUG.WANDB_NAME='Train(CMU0) Test(CMU0ex7)'

