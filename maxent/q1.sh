#!/usr/bin/env bash

dir="/dropbox/17-18/572/maxent/examples/"
trainvec="train2.vectors.txt"
testvec="test2.vectors.txt"
trainvec_out="train2.vectors"
testvec_out="test2.vectors"
model="q1/m1"

mallet import-svmlight --input ${dir}${trainvec} ${dir}${testvec} --output ${trainvec_out} ${testvec_out}
vectors2classify --training-file ${trainvec_out} --testing-file ${testvec_out} --trainer MaxEnt --output-classifier ${model} > me.stdout 2>me.stderr
classifier2info --classifier ${model} > ${model}.txt
