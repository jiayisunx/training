#!/bin/bash

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# runs benchmark and profile
# to use the script:
#   run_and_time_for_bp.sh

pushd pytorch
set -e

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark
set -x
MODE=inference
BATCH_SIZE=4
WARMUP=5
ITERATIONS=${ITERATIONS:-10}
PERFORMANCE_ONLY=False # (True)only test performance, no accuracy, (False) test performance and accuracy. 

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
if [ $MODE = training ]; then
    TOTAL_CORES=`expr $CORES \* $SOCKETS`
    NUM_THREADS=$CORES
else
    TOTAL_CORES=`expr $CORES \* $SOCKETS`
    NUM_THREADS=$CORES
fi

echo "running benchmark"
for i in $(seq 0 $(($TOTAL_CORES / $NUM_THREADS - 1)))
do
echo $i "instance"
export PROFILE=0 #used to control profile, open(1) and close(0)
export USE_MKLDNN=1 #used to enable MKLDNN OP, MKLDNN(1), CPU(0)
export USE_BF16=1 #used to enable bfloat datatype, bfloat16(1), no_bfloat(0)

startid=$(($i*$NUM_THREADS))
endid=$(($i*$NUM_THREADS+$NUM_THREADS-1))
export OMP_SCHEDULE=STATIC OMP_NUM_THREADS=$NUM_THREADS OMP_DISPLAY_ENV=TRUE OMP_PROC_BIND=TRUE GOMP_CPU_AFFINITY="$startid-$endid"  
export OMP_NUM_THREADS=$NUM_THREADS  KMP_AFFINITY=proclist=[$startid-$endid],granularity=fine,explicit

if [ $MODE = training ]; then
    export TRAIN=1
    PROFILE_DIR="./log/train/${i}/"
    time python tools/train_mlperf.py --log ${PROFILE_DIR} --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" \
        DATALOADER.NUM_WORKERS 1 SOLVER.IMS_PER_BATCH $BATCH_SIZE TEST.IMS_PER_BATCH $BATCH_SIZE \
        SOLVER.MAX_ITER ${ITERATIONS} SOLVER.STEPS "(480000, 640000)" SOLVER.BASE_LR 0.0025 MODEL.DEVICE cpu \
        PER_EPOCH_EVAL $PERFORMANCE_ONLY&
else
    export TRAIN=0
    PROFILE_DIR="./log/infer/${i}/"
    time python tools/test_net.py --warmup $WARMUP --iters ${ITERATIONS} --log ${PROFILE_DIR} --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" \
        DATALOADER.NUM_WORKERS 1 SOLVER.MAX_ITER ${ITERATIONS} TEST.IMS_PER_BATCH $BATCH_SIZE MODEL.DEVICE cpu \
        PER_EPOCH_EVAL $PERFORMANCE_ONLY &
fi
done
set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="OBJECT_DETECTION"

echo "RESULT,$result_name,,$result,nvidia,$start_fmt"

popd
