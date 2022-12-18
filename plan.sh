#! /bin/bash -x

# python test/evaluate.py --cfg configs/panoptic/resnet50/save_train_voxel_pred.yaml
python test/evaluate.py --cfg configs/panoptic/resnet50/voxel_g2_b1_32_detail.yaml
python test/evaluate.py --cfg configs/panoptic/resnet50/voxel_g2_b1_64_detail.yaml

# /mnt/blob/jialzhu/workspace/voxel_pose/output/panoptic/multi_person_posenet_50/save_validation_voxel_32_pred/model_best_32.pth.tar

# /mnt/data/jialzhu/workspace/voxel_pose/output/panoptic/multi_person_posenet_50/voxel_g8_b1/final_state.pth.tar


python test/evaluate.py --cfg configs/panoptic/resnet50/std32.yaml TEST.MODEL_FILE='../voxel_g8_b1/model_best.pth.tar' DEBUG.WANDB_KEY='eb29f6f9304c37fa9063f9251f45e703142cedeb' DEBUG.WANDB_NAME=''


config.DATASET.TRAIN_CAM_SEQ = 'CMU0'
config.DATASET.TEST_CAM_SEQ = 'CMU0'
config.DATASET.SAVE_RESULT = None
config.DATASET.DATA_SEQ = 'seq0'



python run/train_3d.py --cfg configs/panoptic/resnet50/voxel_g8_b1.yaml GPUS='0,1' DEBUG.WANDB_KEY='eb29f6f9304c37fa9063f9251f45e703142cedeb' DATASET.TRAIN_CAM_SEQ='CMU1' DATASET.TEST_CAM_SEQ='CMU1' DATASET.CAMERA_NUM=7 DATASET.SAVE_RESULT='voxel_64_pred' DATASET.DATA_SEQ='seq1'


python run/train_3d.py --cfg configs/panoptic/resnet50/std32.yaml 