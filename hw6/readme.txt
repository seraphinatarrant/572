Also included in my tarball are the log from running all 4 lines on condor (stderr has the times) and the accuracy file
that pairs with the desired sys out, in case they are helpful.

Table 1:
Running Time is in Minutes.

+-----------+------+------+---------------+--------------+
| beam_size | topN | topK | Test Accuracy | Running Time |
+-----------+------+------+---------------+--------------+
|         0 |    1 |    1 |       0.96399 |          0.5 |
|         1 |    3 |    5 |       0.96521 |          1.1 |
|         2 |    5 |   10 |       0.96521 |          2.5 |
|         3 |   10 |  100 |       0.96521 |          19.5|
+-----------+------+------+---------------+--------------+



      (\-"""-/)
       |     |
       \ ^ ^ /  .-.
        \_o_/  / /
       /`   `\/  |
      /       \  |
      \ (   ) /  |
     / \_) (_/ \ /
    |   (\-/)   |
    \  --^o^--  /
     \ '.___.' /
    .'  \-=-/  '.
   /   /`   `\   \
  (//./       \.\\)
   `"`         `"`