q1)
Commands:
mallet import-svmlight --input train.vectors.txt test.vectors.txt --output train.vectors test.vectors
vectors2classify --training-file train.vectors --testing-file test.vectors --trainer NaiveBayes > nb.stdout 2>nb.stderr
Results:
NaiveBayesTrainer
Summary. train accuracy mean = 0.9444444444444444 stddev = 0.0 stderr = 0.0
Summary. test accuracy mean = 0.8966666666666666 stddev = 0.0 stderr = 0.0

q2)
+-----------------+--------------------+--------------------+
| cond_prob_delta | Training accuracy  |   Test accuracy    |
+-----------------+--------------------+--------------------+
|             0.1 | 0.92963            |            0.88000 |
|             0.5 | 0.90963            |            0.86000 |
|             1.0 | 0.89519            |            0.83667 |
+-----------------+--------------------+--------------------+

q3)
+-----------------+--------------------+--------------------+
| cond_prob_delta | Training accuracy  |   Test accuracy    |
+-----------------+--------------------+--------------------+
|             0.1 | 0.95741            |            0.91333 |
|             0.5 | 0.95111            |            0.90667 |
|             1.0 | 0.94519            |            0.89667 |
+-----------------+--------------------+--------------------+


    __                      __
 .-'  `'.._...-----..._..-'`  '-.
/                                \
|  ,   ,'                '.   ,  |
 \  '-/                    \-'  /
  '._|          _           |_.'
     |    /\   / \    /\    |
     |    \/   | |    \/    |
      \        \"/         /
       '.    =='^'==     .'
         `'------------'`

