Q1
Commands:
dir="/dropbox/17-18/572/hw5/examples/"
trainvec="train2.vectors.txt"
testvec="test2.vectors.txt"
trainvec_out="train2.vectors"
testvec_out="test2.vectors"
model="q1/m1"

mallet import-svmlight --input ${dir}${trainvec} ${dir}${testvec} --output ${trainvec_out} ${testvec_out}
vectors2classify --training-file ${trainvec_out} --testing-file ${testvec_out} --trainer MaxEnt --output-classifier ${model} > me.stdout 2>me.stderr
classifier2info --classifier ${model} > ${model}.txt

Accuracy:
Summary. train accuracy mean = 0.9685185185185186 stddev = 0.0 stderr = 0.0
Summary. test accuracy mean = 0.8266666666666667 stddev = 0.0 stderr = 0.0

Q2
Testing accuracy=0.82667
Yes, the accuracy is the same as in Q1. This makes sense since I am using the model output in Q1.


Here is a monotreme for you:

                             .
                        ._`-\ )\,`-.-.
                       \'\` \)\ \)\ \|.)
                     \`)  |\)  )\ .)\ )\|
                     \ \)\ |)\  `   \ .')/|
                    ``-.\ \    )\ `  . ., '(
                    \\ -. `)\``- ._ .)` |\(,_
                    `__  '\ `--  _\`. `    (/
                      `\,\       .\\        /
                        '` )  (`-.\\       `
                           /||\    `.  * _*|
                                     `-.( `\
                                         `. \
                                           `0
