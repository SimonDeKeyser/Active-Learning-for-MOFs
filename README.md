**This repository contains the Query by Commitee active learning environment for NequIP**

### Scripts to use:

* init.py &emsp;*Initiate the QbC environment by training N models on an initial dataset*
* train.py &emsp;*Adjust QbC training parameters*
* md.py &emsp;*Adjust QbC MD parameters*

### Starting the first cycle:

* If cp2k = True in train.py, the QbC training will perform MD runs to generate new data, run:
```
bash cycle1.sh
```
from inside the /runs folder

* If cp2k = False in train.py, QbC will be performed on a precalculated MD trajectory (development), run:
```
qsub cycle1.sh -d $(pwd)
```
from inside the /runs folder

### Util scripts: (run from inside QbC folder)

* python util/check.py `cycle number` log &emsp;*Check how the QbC cycle is doing*
* python util/check.py `cycle number` MD &emsp;*After checking, restart unfinished MD runs*
* python util/check.py `cycle number` CP2K &emsp;*After checking, restart unfinished CP2K runs*
