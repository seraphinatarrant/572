#!/bin/sh

deltas="0.1 0.5 1.0"

for delta in $deltas
do
    eval time ./build_NB1.sh train.vectors.txt test.vectors.txt 0.0 $delta outputs/model_file_${delta} outputs/sys_output_${delta} > outputs/acc_file_${delta}
done
