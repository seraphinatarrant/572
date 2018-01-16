Q1
a)
Commands:
mallet import-svmlight --input train.vectors.txt test.vectors.txt --output train.vectors test.vectors
vectors2classify --training-file train.vectors --testing-file test.vectors --trainer DecisionTree > dt.stdout 2>dt.stderr
b)
Summary. train accuracy mean = 0.6377777777777778 stddev = 0.0 stderr = 0.0
Summary. test accuracy mean = 0.5233333333333333 stddev = 0.0 stderr = 0.0

This is the same as the accuracy for DT of depth 4 in following Q2.

Q2
a)
+-------+---------------------+--------------------+
| Depth |  Training Accuracy  |   Test Accuracy    |
+-------+---------------------+--------------------+
|     1 |  0.45296296296296296| 0.4166666666666667 |
|     2 |  0.5207407407407407 | 0.5266666666666666 |
|     4 |  0.6377777777777778 | 0.5233333333333333 |
|    10 |  0.7514814814814815 |                0.6 |
|    20 |  0.8555555555555555 | 0.6833333333333333 |
|    50 |  0.9681481481481482 |                0.7 |
+-------+---------------------+--------------------+
b)
The training accuracy and test accuracy do increase with depth, though by depth of 50 the DT seems to be overfitting,
since the training accuracy jumps to near perfect from depth of 20, and the test accuracy barely changes. So I would
probably stop at 20, based on this.
There are a few other trends (like the way that training always gets better, but test seems to stay the same and
then jump) but that is the salient one.