from distutils.log import set_verbosity
import time
start = time.time()
from dataclasses import dataclass

from pathlib import Path, PosixPath
import logging
import shutil
import os 
import random
import argparse
import datetime as dt
from datetime import timedelta

import ase.io
from nequip.utils import Config

from qbc import qbc

logging.basicConfig(format='',level=logging.INFO)

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
'''
##########################################################################################

root = Path('../../').resolve() # starting the run from /runs folder
head_dir = root / 'qbc_train'
traj_dir = head_dir / 'MD_traj.xyz'                                                                                                                             
n_select = 11
n_val_0 = 1                                                                
n_val_add = 1
max_epochs = 50000   
send_hpc_run = False                                                                 
walltime_per_model_add = dt.timedelta(minutes=10)
load_query_results = False
prop = 'forces'
red = 'mean'
max_index = 3500
cluster = 'accelgor'
env = 'torchenv_stress_accelgor'
cores = '12' # should be 12 when using accelgor
cp2k = False
cp2k_cores = 24
cp2k_walltime = '01:00:00'

##########################################################################################
@dataclass
class qbc_trainer:
    cycle: int
    walltime: timedelta
    root: PosixPath = Path('../../').resolve()
    head_dir: PosixPath = root / 'qbc_train'
    traj_dir: PosixPath = head_dir / 'trajectory.xyz'                                                                                                                             
    n_select: int = 11
    n_val_0: int = 1                                                                
    n_val_add: int = 1
    max_epochs: int = 50000   
    send_hpc_run: bool = False                                                                 
    walltime_per_model_add: timedelta = dt.timedelta(minutes=10)
    load_query_results: bool = False
    prop: str = 'forces'
    red: str = 'mean'
    max_index: int = 3500
    cluster: str = 'accelgor'
    env: str = 'torchenv_stress_accelgor'
    cores: str = '12' # should be 12 when using accelgor
    cp2k: bool = False
    cp2k_cores: int = 24
    cp2k_walltime: str = '01:00:00'
    traj_index: str = ':'
    cp2k_restart: bool = False

    def __post_init__(self):
        logging.info('___ QUERY BY COMMITTEE ___\n')
        logging.info('\t\t###########')
        logging.info('\t\t# CYCLE {} #'.format(cycle))
        logging.info('\t\t###########\n')

        assert self.head_dir.exists(), 'Head directory does not exist'
        assert self.traj_dir.exists(), 'Trajectory path does not exist'
        prev_name = 'cycle{}'.format(cycle-1)
        self.prev_nequip_train_dir = self.head_dir / prev_name / 'results'
        self.prev_dataset_dir = self.head_dir / prev_name / 'data.xyz'
        assert self.prev_nequip_train_dir.is_dir(), 'Previous training directory does not exist'
        assert self.prev_dataset_dir.exists(), 'Previous training dataset directory does not exist'

        self.name = 'cycle{}'.format(self.cycle)
        self.cycle_dir = self.head_dir / self.name
        self.dataset_dir = self.cycle_dir / 'data.xyz'
        self.input_dir = self.cycle_dir / 'new_data.xyz'
        self.output_dir = self.cycle_dir / 'calc_data.xyz'

        if start is None:
            self.start = time.time()
        else:
            self.start = start

    def evaluate_committee(self):
        committee = qbc(name=self.name, models_dir=self.prev_nequip_train_dir, 
                        traj_dir=self.traj_dir, results_dir=self.head_dir,
                        traj_index=self.traj_index, n_select=self.n_select, nequip_train=True
                    )
        assert self.cycle_dir.is_dir(), 'Something went wrong in the qbc class'

        if not self.load_query_results:
            committee.evaluate_committee(save=True)
        else:
            committee.load()

        new_datapoints = committee.select_data(prop=prop, red=red)
        committee.plot_traj_disagreement()

        if self.cp2k:
            ase.io.write(self.input_dir, new_datapoints, format='extxyz')
            if not Path('cycle{}_cp2k'.format(self.cycle)).exists():
                Path('cycle{}_cp2k'.format(self.cycle)).mkdir()
            cp2k_dir = Path('cycle{}_cp2k'.format(self.cycle))
            restart_conf = cp2k_dir / 'restart.yaml'
            with open('cycle{}_cp2k/job.sh'.format(self.cycle),'w') as rsh:
                rsh.write(
                    '#!/bin/sh'
                    '\n\n#PBS -o output.txt'
                    '\n#PBS -e error.txt'
                    '\n#PBS -l walltime={}'
                    '\n#PBS -l nodes=1:ppn={}'
                    '\n\nsource ~/.cp2kenv'
                    '\npython ../../cp2k_main.py {} {} {} {}'.format(self.cp2k_walltime, self.cp2k_cores, str(self.input_dir), str(self.output_dir), self.cp2k_cores, restart_conf)
                )
            config = Config()
            config.cycle = self.cycle
            config.walltime = self.walltime
            config.cores = self.cores
            config.traj_index = self.traj_index
            config.env = self.env
            config.cluster = self.cluster
            config.save(restart_conf, 'yaml')

            os.system('module swap cluster/doduo; cd {}; qsub job.sh -d $(pwd)'.format(cp2k_dir))
        else:  
            ase.io.write(self.output_dir, new_datapoints, format='extxyz')

    def make_config(self):
        config = Config()

        config.root = str(self.nequip_train_dir)
        config.restart = True
        config.append = True

        config.dataset_file_name = str(self.dataset_dir)

        config.wandb_resume = True

        config.n_train = self.dataset_len - self.n_val_0 - self.cycle*self.n_val_add
        config.n_val = self.n_val_0 + self.cycle*self.n_val_add

        config.max_epochs = self.max_epochs
        config.lr_scheduler_name = 'none'

        return config
    
    def run_hpc_job(self, hpc_run_dir, train_dir, config_dir):
        with open(hpc_run_dir / 'run.sh','w') as rsh:
            rsh.write(
                '#!/bin/sh'
                '\n\n#PBS -l walltime={}'
                '\n#PBS -l nodes=1:ppn={}:gpus=1'
                '\n\nsource ~/.{}'
                '\npython ../restart.py {} --update-config {}'.format(str(self.walltime), self.cores, self.env, train_dir, config_dir)
            )

        os.system('module swap cluster/{}; qsub {} -d $(pwd) -e {} -o {}'.format(self.cluster, hpc_run_dir / 'run.sh', hpc_run_dir / 'error', hpc_run_dir / 'output'))

    def restart_training(self):
        new_datapoints = ase.io.read(self.output_dir, index = ':', format='extxyz')
        prev_dataset = ase.io.read(self.prev_dataset_dir, format='extxyz',index=':')
        assert self.n_val_add < self.n_select, 'No training points added but only validation points'
        len_train_add = self.n_select - self.n_val_add
        random.shuffle(new_datapoints)
        new_dataset = new_datapoints[:len_train_add] + prev_dataset + new_datapoints[len_train_add:]
        self.dataset_len = len(new_dataset)

        ase.io.write(self.dataset_dir, new_dataset, format='extxyz')
        logging.info('Saved the new dataset of length {}'.format(len(new_dataset)))

        self.nequip_train_dir = self.cycle_dir / 'results'
        if not self.nequip_train_dir.exists():
            self.nequip_train_dir.mkdir()

        p = Path(self.prev_nequip_train_dir).glob('*')
        model_files = [x for x in p if x.is_dir()]

        self.len_models = 0
        for file in sorted(model_files):
            if 'model' in file.name:
                self.len_models += 1
                if not (self.nequip_train_dir / file.name).exists():
                    shutil.copytree(file, self.nequip_train_dir / file.name)
                    logging.info('Copying NequIP train folder of {}'.format(file.name))

        hpc_dir = self.cycle_dir / 'hpc'
        if not hpc_dir.exists():
            hpc_dir.mkdir()

        if not self.send_hpc_run:
            elapsed_t = time.time()-self.start
            left_t = self.walltime.seconds - elapsed_t
            logging.info('\n## {} hours left over of HPC walltime'.format(round(left_t/3600,3)))
            t_per_model = left_t/self.len_models
            logging.info('## Each of the {} models will train {} hours'.format(self.len_models,round(t_per_model/3600,3)))

        p = Path(self.nequip_train_dir).glob('*')
        model_files = [x for x in p if x.is_dir()]

        for file in sorted(model_files):
            if 'model' in file.name:
                logging.info('\n###################################################################################################\n')
                logging.info('Starting retraining of {}\n'.format(file.name))
                config = self.make_config()

                hpc_run_dir = hpc_dir / file.name
                if not hpc_run_dir.exists():
                    hpc_run_dir.mkdir()
                config.save(str(hpc_run_dir / 'updated_config.yaml'),'yaml')
                config_dir = str(hpc_run_dir / 'updated_config.yaml')
                train_dir = str(file)

                if self.send_hpc_run:
                    self.run_hpc_job(hpc_run_dir, train_dir, config_dir)
                else:
                    os.system('timeout {} python ../restart.py {} --update-config {}'.format(t_per_model-3,train_dir,config_dir))
                    logging.info('\n###################################################################################################')
                    logging.info('\nTotal elapsed time: {} hours of total {} hours'.format(round((time.time()-self.start)/3600,3),round(self.walltime.seconds/3600,3)))
                    
    def start_next_cyle(self):
        old = self.traj_index.split(':')
        old_len = int(old[1]) - int(old[0])
        index = '{}:{}'.format(int(old[0])+old_len, int(old[1])+old_len)        
        next_walltime = self.walltime + self.len_models*self.walltime_per_model_add
        if self.cp2k:
            first_walltime = self.walltime
        else:
            first_walltime = next_walltime

        if int(old[1])+old_len <= self.max_index:
            with open('cycle{}.sh'.format(self.cycle+1),'w') as rsh:
                rsh.write(
                    '#!/bin/sh'
                    '\n\n#PBS -l walltime={}'
                    '\n#PBS -l nodes=1:ppn={}:gpus=1'
                    '\n\nsource ~/.{}'
                    '\npython ../train.py {} {} --traj-index {}'.format(str(first_walltime), self.cores, self.env, self.cycle+1, str(next_walltime), index)
                )
            os.system('module swap cluster/{}; qsub cycle{}.sh -d $(pwd)'.format(self.cluster, self.cycle+1))

if __name__ == "__main__":
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

    Trainer = qbc_trainer(
        cycle = cycle,
        walltime = walltime,
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
        traj_index = traj_index,              
        cp2k_restart = cp2k_restart             
    )

    if cp2k_restart:
        Trainer.restart_training()

    elif (not cp2k_restart) and Trainer.cp2k:
        Trainer.evaluate_committee()

    else:
        Trainer.evaluate_committee()
        Trainer.restart_training()
        if not Trainer.send_hpc_run:
            Trainer.start_next_cyle()