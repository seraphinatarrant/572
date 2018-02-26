#!/usr/bin/env bash
from_dir="examples/"
to_dir_q3="q3/"
to_dir_q4="q4/"

# Q1
time ./calc_emp_exp.sh ${from_dir}train2.vectors.txt ${to_dir_q3}emp_count
# Q2
time ./calc_model_exp.sh ${from_dir}train2.vectors.txt ${to_dir_q4}model_count ${from_dir}m1.txt
time ./calc_model_exp.sh ${from_dir}train2.vectors.txt ${to_dir_q4}model_count2