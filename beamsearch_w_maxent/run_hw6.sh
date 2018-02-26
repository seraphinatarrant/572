#!/usr/bin/env bash

time ./beamsearch_maxent.sh examples/sec19_21.txt examples/sec19_21.boundary examples/m1.txt sys_out_0_1_1 0 1 1 > acc_0_1_1
time ./beamsearch_maxent.sh examples/sec19_21.txt examples/sec19_21.boundary examples/m1.txt sys_out_1_3_5 1 3 5 > acc_1_3_5
time ./beamsearch_maxent.sh examples/sec19_21.txt examples/sec19_21.boundary examples/m1.txt sys_out_2_5_10 2 5 10 > acc_2_5_10
time ./beamsearch_maxent.sh examples/sec19_21.txt examples/sec19_21.boundary examples/m1.txt sys_out_0_1_1 3 10 100 > acc_3_10_100