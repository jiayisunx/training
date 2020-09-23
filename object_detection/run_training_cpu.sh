#!/bin/bash

#Runs benchmark and profile

pushd pytorch
export PROFILE=0 #used to control profile, open(1) and close(0)
export USE_MKLDNN=0 #used to enable MKLDNN OP, MKLDNN(1), CPU(0)

PROFILE_DIR='./log/' #used to save profile log.


### training ###
export TRAIN=1
PROFILE_DIR='./log/train/'
time python tools/train_mlperf.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" --log ${PROFILE_DIR} --iters 200\
       SOLVER.IMS_PER_BATCH 2 TEST.IMS_PER_BATCH 1 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" SOLVER.BASE_LR 0.0025 MODEL.DEVICE cpu

popd
