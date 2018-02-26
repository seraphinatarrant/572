#!/usr/bin/env bash

model_file="model_file"
num_trans="1 5 10 20 50 100 200"

time ./TBL_train.sh examples/train2.txt ${model_file} 1

for num in ${num_trans}
do
    time ./TBL_classify.sh examples/train2.txt ${model_file} sys_output_train_${num} ${num}
    time ./TBL_classify.sh examples/test2.txt ${model_file} sys_output_${num} ${num}
done
