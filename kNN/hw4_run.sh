#!/usr/bin/env bash

time ./build_kNN.sh examples/train.vectors.txt examples/test.vectors.txt 1 1 2sys_output_1_1 > 2acc_file_1_1
time ./build_kNN.sh examples/train.vectors.txt examples/test.vectors.txt 5 1 2sys_output_5_1 > 2acc_file_5_1
time ./build_kNN.sh examples/train.vectors.txt examples/test.vectors.txt 10 1 2sys_output_10_1 > 2acc_file_10_1

time ./build_kNN.sh examples/train.vectors.txt examples/test.vectors.txt 1 2 2sys_output_1_2 > 2acc_file_1_2
time ./build_kNN.sh examples/train.vectors.txt examples/test.vectors.txt 5 2 2sys_output > 2acc_file
time ./build_kNN.sh examples/train.vectors.txt examples/test.vectors.txt 10 2 2sys_output_10_2 > 2acc_file_10_2

cat examples/train.vectors.txt | ./rank_feat_by_chi_square.sh > feat_list