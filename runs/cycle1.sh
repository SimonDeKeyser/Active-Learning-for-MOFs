#!/bin/sh

#PBS -o cycle1_o.txt
#PBS -e cycle1_e.txt
#PBS -l walltime=05:00:00
#PBS -l nodes=1:ppn=12:gpus=1

source ~/.torchenv_stress_accelgor
python ../train.py 1 05:00:00