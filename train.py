from distutils.log import set_verbosity
import time
start_time = time.time()

from pathlib import Path, PosixPath
import argparse
import datetime as dt
from datetime import timedelta

from trainer import QbC_trainer

def args_parse():
    parser = argparse.ArgumentParser(
                description="Perform a Query by Committee cycle."
            )
    parser.add_argument("cycle", help="The number of the QbC cycle ")
    parser.add_argument("walltime", help='The walltime for training the models (should be equal to the HPC walltime for sequential training'
                                        'If send_hpc_run = True: the walltime given to each model training session'
                                        'If send_hpc_run = False: the total walltime for training all models sequentially')
    parser.add_argument("--traj-index", help="The indices of the trajectory to perform QbC on")
    parser.add_argument("--cp2k-restart", help="True: cp2k calculation done, restart training")
    args = parser.parse_args()

    return args

'''
Parameters:
    - head_dir                  Head directory for the QbC environment     
    - traj_dir                  MD trajectory file location
    - n_select                  The number of new datapoints to select
    - n_val_0                   The number of datapoints in the first validation set
    - n_val_add                 The number of datapoints to add to the validation set each cycle (n_val_add < n_select)
    - max_epochs                The maximum amount of epochs performed (these are a summation of all cycles)
    - send_hpc_run              True: train each model seperately by performing a HPC bash command for each
                                False: train each model in one HPC bash command, sequentially
    - walltime_per_model_add    The walltime added per model in each cycle because the dataset increases
    - load_query_results        True: if do_evaluation = True then the disagreement results will be loaded if saved previously
    - prop                      The property to be used in the disagreement metric, choose from: energy, forces, random
    - red                       In the case of prop = forces, choose the reduction from: mean, max
    - cluster                   The HPC cluster to be used; joltik or accelgor
    - env                       The virtual environment that is loaded before each job, together with some modules; torchenv or torchenv_stress_accelgor
    - cores                     The number of HPC cores to be used in a job
    - cp2k                      If True, newly selected datapoints are calculated with cp2k
    - cp2k_cores                The amount of cores used in the cp2K job
    - cp2k_walltime             The walltime the cp2k can use for calculating all new datapoints in a cycle
    - cp2k_qbc_walltime         The walltime for the QbC evaluation, when using cp2k this is done in a first job
'''
##########################################################################################

root = Path('../../').resolve() # starting the run from /runs folder
head_dir = root / 'qbc_train'
traj_dir = head_dir / 'MD_traj.xyz'                                                                                                                             
n_select = 2
n_val_0 = 1                                                                
n_val_add = 1
max_epochs = 50000   
send_hpc_run = False                                                                 
walltime_per_model_add = dt.timedelta(minutes=10)
load_query_results = True
prop = 'forces'
red = 'mean'
max_index = 3500
cluster = 'accelgor'
env = 'torchenv_stress_accelgor'
cores = '12' # should be 12 when using accelgor
cp2k = True
cp2k_cores = 24
cp2k_walltime = '01:00:00'
cp2k_qbc_walltime = '00:10:00'

##########################################################################################

args = args_parse()
cycle = int(args.cycle)
walltime = dt.datetime.strptime(args.walltime, "%H:%M:%S").time()
walltime = dt.timedelta(hours=walltime.hour, minutes=walltime.minute, seconds=walltime.second)
if args.traj_index is None:
    traj_index = ':'
else:
    traj_index = args.traj_index
if args.cp2k_restart is None:
    cp2k_restart = False
else:
    cp2k_restart = args.cp2k_restart

Trainer = QbC_trainer(
    cycle = cycle,
    walltime = walltime,
    start_time = start_time,
    root = root,
    head_dir = head_dir,
    traj_dir = traj_dir,                                                                                                              
    n_select = n_select,
    n_val_0 = n_val_0,                                                           
    n_val_add = n_val_add,              
    max_epochs = max_epochs,              
    send_hpc_run = send_hpc_run,                                                                          
    walltime_per_model_add = walltime_per_model_add,              
    load_query_results = load_query_results,              
    prop = prop,              
    red = red,              
    max_index = max_index,              
    cluster = cluster,              
    env = env,              
    cores = cores,              
    cp2k = cp2k,              
    cp2k_cores = cp2k_cores,              
    cp2k_walltime = cp2k_walltime,
    cp2k_qbc_walltime = cp2k_qbc_walltime,              
    traj_index = traj_index,              
    cp2k_restart = cp2k_restart            
)

if cp2k_restart:
    Trainer.restart_training()
    if not Trainer.send_hpc_run:
        Trainer.start_next_cyle()

elif (not cp2k_restart) and Trainer.cp2k:
    Trainer.evaluate_committee()

else:
    Trainer.evaluate_committee()
    Trainer.restart_training()
    if not Trainer.send_hpc_run:
        Trainer.start_next_cyle()