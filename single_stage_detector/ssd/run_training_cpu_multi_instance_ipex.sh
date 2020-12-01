#!/bin/bash
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

ARGS=""

if [[ "$1" == "dnnl" ]]
then
    ARGS="$ARGS --dnnl"
    echo "### running auto_dnnl mode"
fi

if [[ "$2" == "bf16" ]]
then
    ARGS="$ARGS --mix-precision"
    echo "### running bf16 datatype"
fi

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

# change this number to adjust number of instances
CORES_PER_INSTANCE=$CORES

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

export OMP_NUM_THREADS=$CORES_PER_INSTANCE
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$CORES_PER_INSTANCE"
echo -e "### using $KMP_SETTING\n\n"
sleep 3

INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
LAST_INSTANCE=`expr $INSTANCES - 1`
INSTANCES_PER_SOCKET=`expr $INSTANCES / $SOCKETS`
for i in $(seq 1 $LAST_INSTANCE); do
    numa_node_i=`expr $i / $INSTANCES_PER_SOCKET`
    start_core_i=`expr $i \* $CORES_PER_INSTANCE`
    end_core_i=`expr $start_core_i + $CORES_PER_INSTANCE - 1`
    LOG_i=training_ipex_ins${i}.txt

    echo "### running on instance $i, numa node $numa_node_i, core list {$start_core_i, $end_core_i}..."
    numactl --physcpubind=$start_core_i-$end_core_i --membind=$numa_node_i python train.py \
        --epochs 1 --warmup-factor 0 --lr=2.5e-3 --no-save --threshold=0.23 \
        --data /lustre/dataset/COCO2017 --skip-test -i 100 --ipex $ARGS 2>&1 | tee $LOG_i &
done

numa_node_0=0
start_core_0=0
end_core_0=`expr $CORES_PER_INSTANCE - 1`
LOG_0=training_ipex_ins0.txt

echo "### running on instance 0, numa node $numa_node_0, core list {$start_core_0, $end_core_0}...\n\n"
numactl --physcpubind=$start_core_0-$end_core_0 --membind=$numa_node_0 python train.py \
        --epochs 1 --warmup-factor 0 --lr=2.5e-3 --no-save --threshold=0.23 \
        --data /lustre/dataset/COCO2017 --skip-test -i 100 --ipex $ARGS 2>&1 | tee $LOG_0
