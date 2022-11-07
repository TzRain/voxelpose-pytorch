#! /bin/bash -x

python test/evaluate.py --cfg configs/panoptic/resnet50/save_voxel_pred.yaml
python test/evaluate.py --cfg configs/panoptic/resnet50/validation_save_voxel_pred.yaml