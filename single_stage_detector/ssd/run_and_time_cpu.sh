#!/bin/bash
# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time_cpu.sh

set -e

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark
set -x
MODE="inference"
BATCH_SIZE=1
THRESHOLD=0.23
PERF_PRERUN_WARMUP=5
NUMEPOCHS=${NUMEPOCHS:-1}
LR=${LR:-"2.5e-3"}
TOTLE_ITERATIONS=${TOTLE_ITERATIONS:-10}
ARCH="ssd300"
DUMMY=0 # (0)real data,(1)dummy data

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
if [ $MODE = training ]; then
    TOTAL_CORES=`expr $CORES \* $SOCKETS`
    NUM_THREADS=$CORES
else
    TOTAL_CORES=`expr $CORES \* $SOCKETS`
    NUM_THREADS=$CORES
fi

PROFILE_DIR='./log_instance'
rm -rf ${PROFILE_DIR}
mkdir ${PROFILE_DIR}


echo "running benchmark"
for i in $(seq 0 $(($TOTAL_CORES / $NUM_THREADS - 1)))
do
echo $i "inference"
startid=$(($i*$NUM_THREADS))
endid=$(($i*$NUM_THREADS+$NUM_THREADS-1))
export OMP_SCHEDULE=STATIC OMP_NUM_THREADS=$NUM_THREADS OMP_DISPLAY_ENV=TRUE OMP_PROC_BIND=TRUE GOMP_CPU_AFFINITY="$startid-$endid"  
export OMP_NUM_THREADS=$NUM_THREADS  KMP_AFFINITY=proclist=[$startid-$endid],granularity=fine,explicit
export DATASET_DIR="/lustre/dataset/COCO2017"
# export TORCH_MODEL_ZOO="/data/torchvision"
export USE_MKLDNN=1
#export USE_JIT=1
#export PROFILE=1   # profiling for the whole iteration
#export PROFILE_ITER=1  # profiling for each iteration
# export MKLDNN_VERBOSE=1

python performance_test.py \
  -a $ARCH \
  --epochs "${NUMEPOCHS}" \
  --warmup-factor 0 \
  --lr "${LR}" \
  --no-save \
  --threshold=$THRESHOLD \
  --data ${DATASET_DIR} \
  --no-cuda \
  -b $BATCH_SIZE \
  --totle-iteration "${TOTLE_ITERATIONS}" \
  -m $MODE \
  --perf-prerun-warmup $PERF_PRERUN_WARMUP \
  --log ${PROFILE_DIR} \
  --dummy $DUMMY &
ret_code=$?
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
