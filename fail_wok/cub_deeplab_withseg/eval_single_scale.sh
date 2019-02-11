#!/bin/sh
exp=res-psp101_coarse_shuffle
EXP_DIR=exp/drivable/$exp
mkdir -p ${EXP_DIR}/result
now=$(date +"%Y%m%d_%H%M%S")
cp eval.py ${EXP_DIR}

part=Segmentation
#part=Test
numGPU=1
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 srun --mpi=pmi2 --gres=gpu:$numGPU -n1 --ntasks-per-node=$numGPU --partition=$part --job-name=${exp} --kill-on-bad-exit=1 \
python eval.py \
  --data_root= \
  --val_list=/mnt/lustre/dingmingyu/Research/weakly_supervised/CUB_DATA/att_list.txt \
  --split=val \
  --layers=50 \
  --classes=2 \
  --backbone=resnet \
  --net_type=0 \
  --crop_h=441 \
  --crop_w=441 \
  --zoom_factor=2 \
  --ignore_label=255 \
  --scales 1.0 \
  --has_prediction=0 \
  --gpu 0 \
  --model_path=exp/drivable/res101_psp_coarse/model/train_epoch_50.pth \
  --save_folder=${EXP_DIR}/result/epoch_50/val_scale/ms \
  --colors_path=data/cityscapes/cityscapescolors.txt \
  --names_path=data/cityscapes/cityscapesnames.txt \
  2>&1 | tee ${EXP_DIR}/result/epoch_50-val-ms-$now.log

# --scales 0.5 0.75 1.0 1.25 1.5 1.75 \
# --scales 1.0 \
#  --val_list=/mnt/lustre/share/dingmingyu/cityscapes/list/fine_val.txt \
