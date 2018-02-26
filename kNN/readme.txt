Table 1:
So, I played around with shuffling the training data, and noticed that cosine accuracy stayed (more) constant than
Euclidean did. It isn't tie-breaking, since that is alphabetical. So the only reason I could think of (after debugging
my code extensively) was that, if there are some num of vectors that are the same distance from the target vectors,
there could be multiple equally valid combinations that make up k. e.g. if k=5, and there are 4 vectors at distance 1
and 6 vectors at distance 2, then there are 6 different valid combos for k.
I decided to try to implement two different sorting algorithms to see if this made a difference, and indeed it does.
However, I don't immediately understand why.
Can you tell me why a heapq and a partition sort would do this differently?

Note that I submitted the partition sort version (as I guessed that would be a more standard way to handle it),
but I can submit the other if you like, as it's a one line code change.

With heapq sort:
+----+--------------------+-----------------+
| k  | Euclidean distance | Cosine distance |
+----+--------------------+-----------------+
|  1 |            0.63667 |         0.72000 |
|  5 |            0.65000 |         0.70667 |
| 10 |            0.64000 |         0.66667 |
+----+--------------------+-----------------+


With partition sort:
+----+--------------------+-----------------+
| k  | Euclidean distance | Cosine distance |
+----+--------------------+-----------------+
|  1 |            0.61667 |         0.72000 |
|  5 |            0.65333 |         0.69000 |
| 10 |            0.63333 |         0.66667 |
+----+--------------------+-----------------+



       :     :
        __    |     |    _,_
       (  ~~^-l_____],.-~  /
        \    ")\ "^k. (_,-"
         `>._  ' _ `\  \
      _.-~/'^k. (0)  ` (0
   .-~   {    ~` ~    ..T
  /   .   "-..       _.-'
 /    Y        .   "T
Y     l         ~-./l_
|      \          . .<'
|       `-.._  __,/"r'
l   .-~~"-.    /    I
 Y         Y "~[    |
  \         \_.^--, [
   \            _~> |
    \      ___)--~  |
     ^.       :     l
       ^.   _.j     |
         Y    I     |
         l    l     I
          Y    \    |
           \    ^.  |
            \     ~-^.
             ^.       ~"--.,_
              |~-._          ~-.
              |    ~Y--.,_      ^.
              :     :     "x      \
                            \      \.
                             \      ]
                              ^._  .^
                                 ~~