#!/usr/bin/env bash

if [ $# -ge 3 ]
then
    python3 calc_model_exp.py $1 $2 -m $3
else
    python3 calc_model_exp.py $@
fi