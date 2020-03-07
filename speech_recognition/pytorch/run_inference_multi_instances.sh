#!/bin/bash

NUM_SOCKET=`lscpu | grep 'Socket(s)' | awk '{print $NF}'`
NUM_NUMA_NODE=`lscpu | grep 'NUMA node(s)' | awk '{print $NF}'`
CORES_PER_SOCKET=`lscpu | grep 'Core(s) per socket' | awk '{print $NF}'`
TOTAL_CORES=$((CORES_PER_SOCKET * NUM_SOCKET))
CORES_PER_NUMA=$((TOTAL_CORES / NUM_NUMA_NODE))
echo "target machine has $TOTAL_CORES physical core(s) on $NUM_NUMA_NODE numa nodes of $NUM_SOCKET socket(s)."


KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"


export $KMP_SETTING
export OMP_NUM_THREADS=$CORES_PER_NUMA

echo -e "### using OMP_NUM_THREADS=$CORES_PER_NUMA"
echo -e "### using $KMP_SETTING\n\n"
sleep 3

INSTANCES=`expr $TOTAL_CORES / $CORES_PER_NUMA`
LAST_INSTANCE=`expr $INSTANCES - 1`
INSTANCES_PER_SOCKET=`expr $INSTANCES / $NUM_SOCKET`

# model training configs
RANDOM_SEED=1
MODEL_PATH='models/deepspeech_10.pth'
BATCH_SIZE=1
LAST_INSTANCE=0

for i in $(seq 0 $LAST_INSTANCE); 
do
    numa_node_i=`expr $i / $INSTANCES_PER_SOCKET`
    start_core_i=`expr $i \* $CORES_PER_NUMA`
    end_core_i=`expr $start_core_i + $CORES_PER_NUMA - 1`
    LOG_i=inference_cpu_bs${BATCH_SIZE}_ins${i}.txt

    echo "### running on instance $i, numa node $numa_node_i, core list {$start_core_i, $end_core_i}..."
    numactl --physcpubind=$start_core_i-$end_core_i --membind=$numa_node_i python -u inference.py --seed $RANDOM_SEED \
                                                                                                  --model_path $MODEL_PATH \
                                                                                                  --batch_size $BATCH_SIZE \
                                                                                                  2>&1 | tee $LOG_i &
done
wait

# numa_node_0=0
# start_core_0=0
# end_core_0=`expr $CORES_PER_INSTANCE - 1`
# LOG_0=inference_cpu_bs${BATCH_SIZE}_ins0.txt

# echo "### running on instance 0, numa node $numa_node_0, core list {$start_core_0, $end_core_0}...\n\n"
# numactl --physcpubind=$start_core_0-$end_core_0 --membind=$numa_node_0 python -u main.py -e UCF101 \
#     --batch-size-eval $BATCH_SIZE \
#     --no-cuda \
#     2>&1 | tee $LOG_0

# sleep 10
# echo -e "\n\n Sum sentences/s together:"
# for i in $(seq 0 $LAST_INSTANCE); do
#     log=inference_cpu_bs${BATCH_SIZE}_ins${i}.txt
#     tail -n 2 $log
# done