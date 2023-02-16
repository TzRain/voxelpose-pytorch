#! /bin/bash -x
python test/evaluate.py --cfg configs/panoptic/resnet50/multi-cam/std64_CMU4_12-19.yaml GPUS='0,1,2,3' DATASET.TRAIN_CAM_SEQ='CMU0' DATASET.TEST_CAM_SEQ='CMU0' DATASET.SAVE_RESULT='Train(CMU4)_Test(CMU0)_02-09' DATASET.CAMERA_NUM=5 DEBUG.WANDB_NAME='Train(CMU4) Test(CMU3)'

python test/evaluate.py --cfg configs/panoptic/resnet50/multi-cam/std64_CMU4_12-19.yaml GPUS='0,1,2,3' DATASET.TRAIN_CAM_SEQ='CMU1' DATASET.TEST_CAM_SEQ='CMU1' DATASET.SAVE_RESULT='Train(CMU4)_Test(CMU1)_02-09' DATASET.CAMERA_NUM=7 DEBUG.WANDB_NAME='Train(CMU4) Test(CMU1)'

python test/evaluate.py --cfg configs/panoptic/resnet50/multi-cam/std64_CMU4_12-19.yaml GPUS='0,1,2,3' DATASET.TRAIN_CAM_SEQ='CMU2' DATASET.TEST_CAM_SEQ='CMU2' DATASET.SAVE_RESULT='Train(CMU4)_Test(CMU2)_02-09' DATASET.CAMERA_NUM=7 DEBUG.WANDB_NAME='Train(CMU4) Test(CMU2)'

# python test/evaluate.py --cfg configs/panoptic/resnet50/multi-cam/std64_CMU4_12-19.yaml GPUS='0,1,2,3' DATASET.TRAIN_CAM_SEQ='CMU3' DATASET.TEST_CAM_SEQ='CMU3' DATASET.SAVE_RESULT='Train(CMU4)_Test(CMU3)_02-09' DATASET.CAMERA_NUM=4 DEBUG.WANDB_NAME='Train(CMU4) Test(CMU3)'

# python test/evaluate.py --cfg configs/panoptic/resnet50/multi-cam/std64_CMU4_12-19.yaml GPUS='0,1,2,3' DATASET.TRAIN_CAM_SEQ='CMU4' DATASET.TEST_CAM_SEQ='CMU4' DATASET.SAVE_RESULT='Train(CMU4)_Test(CMU4)_02-09' DATASET.CAMERA_NUM=10 DEBUG.WANDB_NAME='Train(CMU4) Test(CMU4)'


python test/evaluate.py --cfg configs/panoptic/resnet50/multi-cam/std64_CMU4_12-19.yaml GPUS='0,1,2,3' DATASET.TRAIN_CAM_SEQ='CMU2' DATASET.TEST_CAM_SEQ='CMU2' DATASET.SAVE_RESULT='Train(CMU4)_Test(CMU2)_02-09' DATASET.CAMERA_NUM=7 DEBUG.WANDB_NAME='Train(CMU0)'


