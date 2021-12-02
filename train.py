import time
start = time.time()

from pathlib import Path
import logging
import shutil
import os 
import random
import argparse

import ase.io
from nequip.utils import Config

from qbc import qbc

logging.basicConfig(format='',level=logging.INFO)
parser = argparse.ArgumentParser(
            description="Perform a Query by Committee cycle."
        )
parser.add_argument("cycle", help="The number of the QbC cycle ")
parser.add_argument("--traj-index", help="The indices of the trajectory to perform QbC on")
args = parser.parse_args()
cycle = int(args.cycle)
if args.traj_index:
    traj_index = args.traj_index
else:
    traj_index = ':'

'''
Parameters:
    - head_dir              Head directory for the QbC environment     
    - traj_dir              MD trajectory file location
    - n_select              The number of new datapoints to select
    - n_val_0               The number of datapoints in the first validation set
    - n_val_add             The number of datapoints to add to the validation set each cycle (n_val_add < n_select)
    - max_epochs            The maximum amount of epochs performed (these are a summation of all cycles)
    - send_hpc_run          True: train each model seperately by performing a HPC bash command for each
                            False: train each model in one HPC bash command, sequentially
    - walltime              If send_hpc_run = True: the walltime given to each model's training session
                            If send_hpc_run = False: the total walltime for training all models sequentially
    - do_evaluation         True: do the query by committee
                            False: only possible if the new dataset is already saved as data.xyz in the current cycle folder
    - load_query_results    True: if do_evaluation = True then the disagreement results will be loaded if saved previously
    - prev_dataset_len      Length of the previous dataset, only has to be given if do_query = False
'''
##########################################################################################

head_dir = Path('/scratch/gent/vo/000/gvo00003/vsc43785/Thesis/q4/qbc_train') 
traj_dir = head_dir / 'trajectory.xyz'                                                                                                                             
n_select = 100
n_val_0 = 10                                                                 
n_val_add = 10
max_epochs = 50000   
send_hpc_run = False                                                                    
walltime = '04'
do_evaluation = True
load_query_results = False
prev_dataset_len = 1050

##########################################################################################
logging.info('___ QUERY BY COMMITTEE ___\n')
logging.info('\t\t###########')
logging.info('\t\t# CYCLE {} #'.format(cycle))
logging.info('\t\t###########\n')

assert head_dir.exists(), 'Head directory does not exist'
assert traj_dir.exists(), 'Trajectory path does not exist'
prev_name = 'cycle{}'.format(cycle-1)
prev_nequip_train_dir = head_dir / prev_name / 'results'
prev_dataset_dir = head_dir / prev_name / 'data.xyz'
assert prev_nequip_train_dir.is_dir(), 'Previous training directory does not exist'
assert prev_dataset_dir.exists(), 'Previous training dataset directory does not exist'

name = 'cycle{}'.format(cycle)
cycle_dir = head_dir / name
dataset_dir = cycle_dir / 'data.xyz'

def evaluate_committee(load):
    committee = qbc(name=name, models_dir=prev_nequip_train_dir, 
                    traj_dir=traj_dir, results_dir=head_dir,
                    traj_index=traj_index, n_select=n_select, nequip_train=True
                )
    assert cycle_dir.is_dir(), 'Something went wrong in the qbc class'

    if not load:
        committee.evaluate_committee(save=True)
    else:
        committee.load()

    new_datapoints = committee.select_data('forces','mean')
    committee.plot_traj_disagreement()
    prev_dataset = ase.io.read(prev_dataset_dir,format='extxyz',index=':')
    assert n_val_add < n_select, 'No training points added but only validation points'
    len_train_add = n_select - n_val_add
    random.shuffle(new_datapoints)
    new_dataset = new_datapoints[:len_train_add] + prev_dataset + new_datapoints[len_train_add:]
    dataset_len = len(new_dataset)

    ase.io.write(dataset_dir,new_dataset,format='extxyz')
    logging.info('Saved the new dataset of length {}'.format(len(new_dataset)))

    return dataset_len

if do_evaluation:
    dataset_len = evaluate_committee(load_query_results)
else:
    dataset_len = prev_dataset_len + n_select

assert dataset_dir.exists(), 'The dataset was not saved yet, do_query first'
nequip_train_dir = cycle_dir / 'results'
if not nequip_train_dir.exists():
    nequip_train_dir.mkdir()

p = Path(prev_nequip_train_dir).glob('**/*')
model_files = [x for x in p if x.is_dir()]

len_models = 0
for file in sorted(model_files):
    if not file.name == 'processed':
        len_models += 1
        if not (nequip_train_dir / file.name).exists():
            shutil.copytree(file, nequip_train_dir / file.name)
            logging.info('Copying NequIP train folder of {}'.format(file.name))

def make_config():
    config = Config()

    config.root = str(nequip_train_dir)
    config.restart = True
    config.append = True

    config.dataset_file_name = str(dataset_dir)

    config.wandb_resume = True

    config.n_train = dataset_len - n_val_0 - cycle*n_val_add
    config.n_val = n_val_0 + cycle*n_val_add

    config.max_epochs = max_epochs

    return config

hpc_dir = cycle_dir / 'hpc'
if not hpc_dir.exists():
    hpc_dir.mkdir()

if not send_hpc_run:
    elapsed_t = time.time()-start
    left_t = int(walltime)*3600 - elapsed_t
    logging.info('\n## {} hours left over of HPC walltime'.format(round(left_t/3600,3)))
    t_per_model = left_t/len_models
    logging.info('## Each of the {} models will train {} hours'.format(len_models,round(t_per_model/3600,3)))

def run_hpc(hpc_run_dir, train_dir, config_dir):
    with open(hpc_run_dir / 'run.sh','w') as rsh:
        rsh.write(
            '#!/bin/sh'
            '\n\n#PBS -l walltime={}:00:00'
            '\n#PBS -l nodes=1:ppn=8:gpus=1'
            '\n\nsource ~/.torchenv'
            '\npython ../restart.py {} --update-config {}'.format(walltime,train_dir,config_dir)
        )

    os.system('qsub {} -d $(pwd) -e {} -o {}'.format(hpc_run_dir / 'run.sh', hpc_run_dir / 'error', hpc_run_dir / 'output'))

p = Path(nequip_train_dir).glob('**/*')
model_files = [x for x in p if x.is_dir()]

for file in sorted(model_files):
    if not file.name == 'processed':
        logging.info('\n###################################################################################################\n')
        logging.info('Starting retraining of {}\n'.format(file.name))
        config = make_config()

        hpc_run_dir = hpc_dir / file.name
        if not hpc_run_dir.exists():
            hpc_run_dir.mkdir()
        config.save(str(hpc_run_dir / 'updated_config.yaml'),'yaml')
        config_dir = str(hpc_run_dir / 'updated_config.yaml')
        train_dir = str(file)

        if send_hpc_run:
            run_hpc(hpc_run_dir, train_dir, config_dir)
        else:
            os.system('timeout {} python ../restart.py {} --update-config {}'.format(t_per_model-3,train_dir,config_dir))
            logging.info('\n###################################################################################################')
            logging.info('\nTotal elapsed time: {} hours of total {} hours'.format(round((time.time()-start)/3600,3),round(int(walltime),3)))
            
if not send_hpc_run:
    old = traj_index.split(':')
    old_len = int(old[1]) - int(old[0])
    index = '{}:{}'.format(int(old[0])+old_len, int(old[1])+old_len)
    with open('cycle{}.sh'.format(cycle+1),'w') as rsh:
        rsh.write(
            '#!/bin/sh'
            '\n\n#PBS -l walltime={}:00:00'
            '\n#PBS -l nodes=1:ppn=8:gpus=1'
            '\n\nsource ~/.torchenv'
            '\npython ../train.py {} --traj-index {}'.format(walltime,cycle+1,index)
        )
    if int(old[1])+old_len <= 30000:
        os.system('qsub cycle{}.sh -d $(pwd)'.format(cycle+1))