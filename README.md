**This repository contains the Query by Commitee active learning environment for NequIP**

#Scripts to use:**

* init.py           ## Initiate the QbC environment by training N models on an initial dataset
* train.py          ## Adjust QbC training parameters
* md.py             ## Adjust QbC MD parameters
* /runs/cycle1.sh   ## Run this bash script from inside the runs folder to start cycle 1

##Util scripts: (run from inside QbC folder)

* python util/check.py <cycle number> log   ## Check how the QbC cycle is doing
* python util/check.py <cycle number> MD    ## After checking, restart unfinished MD runs
* python util/check.py <cycle number> CP2K  ## After checking, restart unfinished CP2K runs
