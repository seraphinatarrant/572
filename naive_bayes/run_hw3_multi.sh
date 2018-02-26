#!/bin/sh

deltas="0.1 0.5 1.0"

for delta in $deltas
do
    eval time ./build_NB2.sh train.vectors.txt test.vectors.txt 0.0 $delta outputs/model_file_multi${delta} outputs/sys_output_multi${delta} > outputs/acc_file_multi${delta}
done
