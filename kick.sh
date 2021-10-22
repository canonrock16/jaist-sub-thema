#!/bin/bash
DATE=$(date '+%Y-%m-%d_%H:%M:%S')
DATA_SIZE=1M

# for i in 16 32 64 96 128; do # KAGAYAKI上での並列数
# for i in 2 4; do       # ローカルでの並列数
for i in 4; do
    # for j in 1 2 3; do # 反復して平均と標準偏差を計算
    for j in 1; do # とりあえず動くか見る用
        mpiexec -n $i --allow-run-as-root python paralleled_collaborative_filtering_send.py >./logs/send/log_send_parallel_${i}_${DATA_SIZE}_${DATE}_${j}.txt
        mpiexec -n $i --allow-run-as-root python paralleled_collaborative_filtering_isend.py >./logs/isend/log_isend_parallel_${i}_${DATA_SIZE}_${DATE}_${j}.txt
        mpiexec -n $i --allow-run-as-root python paralleled_collaborative_filtering_scatter.py >./logs/scatter/log_scatter_parallel_${i}_${DATA_SIZE}_${DATE}_${j}.txt
    done
done

# 256並列
# for j in 1 2 3; do # 反復して平均と標準偏差を計算
# mpiexec --hostfile ${PBS_NODEFILE} -n 128 --allow-run-as-root python paralleled_collaborative_filtering_send.py >./logs/send/log_send_parallel_256_${DATA_SIZE}_${DATE}_${j}.txt
# mpiexec --hostfile ${PBS_NODEFILE} -n 128 --allow-run-as-root python paralleled_collaborative_filtering_isend.py >./logs/isend/log_isend_parallel_256_${DATA_SIZE}_${DATE}_${j}.txt
# mpiexec --hostfile ${PBS_NODEFILE} -n 128 --allow-run-as-root python paralleled_collaborative_filtering_scatter.py >./logs/scatter/log_scatter_parallel_256_${DATA_SIZE}_${DATE}_${j}.txt
# done
