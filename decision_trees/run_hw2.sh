#!/bin/sh

depths="1 2 4 10 20 50"
mingains="0 0.1"

for gain in $mingains
do
    for depth in $depths
    do
        echo "gain $gain depth $depth\n"
        eval time ./build_dt.sh examples/train.vectors.txt examples/test.vectors.txt $depth $gain outputs/model_file_${depth}_${gain} outputs/sys_output_${depth}_${gain} > outputs/acc_file_${depth}_${gain}
    done
done