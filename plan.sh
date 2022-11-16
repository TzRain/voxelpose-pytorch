#! /bin/bash -x

# python test/evaluate.py --cfg configs/panoptic/resnet50/save_train_voxel_pred.yaml
python test/evaluate.py --cfg configs/panoptic/resnet50/voxel_g2_b1_32.yaml
python test/evaluate.py --cfg configs/panoptic/resnet50/voxel_g2_b1_64.yaml

# /mnt/blob/jialzhu/workspace/voxel_pose/output/panoptic/multi_person_posenet_50/save_validation_voxel_32_pred/model_best_32.pth.tar

# /mnt/data/jialzhu/workspace/voxel_pose/output/panoptic/multi_person_posenet_50/voxel_g8_b1/final_state.pth.tar