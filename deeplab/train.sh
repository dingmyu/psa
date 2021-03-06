#!/bin/sh
exp=res101_psp_coarse
EXP_DIR=exp/drivable/$exp
mkdir -p ${EXP_DIR}/model
now=$(date +"%Y%m%d_%H%M%S")
cp train.sh train.py ${EXP_DIR}
#part=Segmentation
part=Segmentation
numGPU=4
nodeGPU=4
GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p $part --gres=gpu:$nodeGPU -n$numGPU --ntasks-per-node=$nodeGPU --job-name=${exp} \
python -u train.py \
  --layers=50 \
  --backbone=resnet \
  --port=12345 \
  --syncbn=1 \
  --classes=20 \
  --crop_h=448 \
  --crop_w=448 \
  --zoom_factor=1 \
  --base_lr=1e-2 \
  --epochs=30 \
  --start_epoch=1 \
  --batch_size=16 \
  --bn_group=4 \
  --save_step=1 \
  --save_path=${EXP_DIR}/model \
  --evaluate=0 \
  --ignore_label 255 \
  --workers 2 \
  2>&1 | tee ${EXP_DIR}/model/train-$now.log
