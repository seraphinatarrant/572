Table 1:
+-----+-------------------+---------------+
|  N  | Training Accuracy | Test Accuracy |
+-----+-------------------+---------------+
|   1 |           0.45296 |       0.41667 |
|   5 |           0.61481 |       0.63667 |
|  10 |           0.68407 |       0.69667 |
|  20 |           0.75296 |       0.72667 |
|  50 |           0.83741 |       0.76333 |
| 100 |           0.89519 |       0.80000 |
| 200 |           0.94593 |       0.74000 |
+-----+-------------------+---------------+

For both train and test the baseline annotator accuracy is 33% (all three labels are evenly distributed).
So even just 5 transformations doubles the baseline accuracy, which is pretty cool!
We start to fit to noise at possibly around 20 transformations, and definitely between 100 and 200 transformations.

I have run out of marsupials...so here is a badger.

               ___,,___
          _,-='=- =-  -`"--.__,,.._
       ,-;// /  - -       -   -= - "=.
     ,'///    -     -   -   =  - ==-=\`.
    |/// /  =    `. - =   == - =.=_,,._ `=/|
   ///    -   -    \  - - = ,ndDMHHMM/\b  \\
 ,' - / /        / /\ =  - /MM(,,._`YQMML  `|
<_,=^Kkm / / / / ///H|wnWWdMKKK#""-;. `"0\  |
       `""QkmmmmmnWMMM\""WHMKKMM\   `--. \> \
             `""'  `->>>    ``WHMb,.    `-_<@)
                               `"QMM`.
                                  `>>>