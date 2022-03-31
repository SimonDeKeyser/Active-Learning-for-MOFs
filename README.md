**This repository contains the Query by Commitee active learning environment for NequIP**

### Scripts to use:

* init.py &emsp;*Initiate the QbC environment by training N models on an initial dataset*
* train.py &emsp;*Adjust QbC training parameters*
* md.py &emsp;*Adjust QbC MD parameters*
* /runs/cycle1.sh &emsp;*Run this bash script from inside the runs folder to start cycle 1*

### Util scripts: (run from inside QbC folder)

* python util/check.py <cycle number> log &emsp;*Check how the QbC cycle is doing*
* python util/check.py <cycle number> MD &emsp;*After checking, restart unfinished MD runs*
* python util/check.py <cycle number> CP2K &emsp;*After checking, restart unfinished CP2K runs*
