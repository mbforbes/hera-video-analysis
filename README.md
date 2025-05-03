plan

-  [x] extract
   -  [x] pop
   -  [x] idles
   -  [x] time
   -  [x] date
   -  [x] maybe players / score
   -  [x] whoops way more lol
-  [x] make robust to bad frames (beg/end of video)
-  [x] save metadata
-  [x] track cost
-  [x] ensure frame shape expected
-  [x] clean up code
-  [x] save as known data structure if possible
-  [x] use frame's cached result if it exists
-  [ ] try binary search for age transitions
   -  can just do vanilla binary search, but track frame states, age 0 (dark) = look right, age 1 (feudal) = look left, stop when consecutive frames known different, need one search per boundary i think
