#!/bin/sh

#PBS -o output.txt
#PBS -e error.txt
#PBS -l walltime=01:30:00
#PBS -l nodes=1:ppn=12

source ~/.torchenv_stress_accelgor
python check.py 7 eval --plot True