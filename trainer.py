from distutils.log import set_verbosity
import time
start = time.time()
from dataclasses import dataclass

from pathlib import Path, PosixPath
import logging
import shutil
import os 
import random
import datetime as dt
from datetime import timedelta

import ase.io
from nequip.utils import Config
import pandas as pd

try:
    import ssh_keys
except ImportError:
    pass 
from vsc_shell import VSC_shell

from qbc import QbC, CNNP, Attacker

logging.basicConfig(format='',level=logging.INFO)

@dataclass
class QbC_trainer:
    cycle: int
    walltime: timedelta
    start_time: float
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
    cp2k_qbc_walltime: str = '00:10:00'
    traj_index: str = ':'
    cp2k_restart: bool = False
    cp2k_cluster: str = 'doduo'

    def __post_init__(self):
        logging.info('___ QUERY BY COMMITTEE ___\n')
        logging.info('\t\t###########')
        logging.info('\t\t# CYCLE {} #'.format(self.cycle))
        logging.info('\t\t###########\n')

        assert self.head_dir.exists(), 'Head directory does not exist'
        assert self.traj_dir.exists(), 'Trajectory path does not exist'
        prev_name = 'cycle{}'.format(self.cycle-1)
        self.prev_nequip_train_dir = self.head_dir / prev_name / 'results'
        self.prev_dataset_dir = self.head_dir / prev_name / 'data.xyz'
        assert self.prev_nequip_train_dir.is_dir(), 'Previous training directory does not exist'
        assert self.prev_dataset_dir.exists(), 'Previous training dataset directory does not exist'

        self.name = 'cycle{}'.format(self.cycle)
        self.cycle_dir = self.head_dir / self.name
        self.dataset_dir = self.cycle_dir / 'data.xyz'
        self.input_dir = self.cycle_dir / 'new_data.xyz'
        self.output_dir = self.cycle_dir / 'calc_data.xyz'

    def adverserial_attack(self):
        cnnp = CNNP(self.prev_nequip_train_dir).load_models_from_nequip_training(attack=True)
        committee = Attacker(name=self.name, cnnp=cnnp, dataset_dir=self.prev_dataset_dir, results_dir=self.head_dir,
                        n_select=self.n_select
                        )
        results = committee.attack(100)
        df = pd.DataFrame(results)

        traj = [committee.initial_ase.copy()]
        at = committee.initial_ase.copy()
        volumes = [152*at.get_volume()/(2*len(at))]
        for transl in df.delta:
            at = committee.initial_ase.copy()
            at.translate(transl)
            traj.append(at)
            volumes.append(152*at.get_volume()/(2*len(at)))
        committee.visualise_attack(df, volumes)
        #ase.io.write('cmp.xyz', traj, format='extxyz')

    def query(self):
        cnnp = CNNP(self.prev_nequip_train_dir).load_models_from_nequip_training()
        committee = QbC(name=self.name, cnnp=cnnp, traj_dir=self.traj_dir, results_dir=self.head_dir,
                        traj_index=self.traj_index, n_select=self.n_select
                        )
        assert self.cycle_dir.is_dir(), 'Something went wrong in the qbc class'

        if not self.load_query_results:
            committee.evaluate_committee(save=True)
        else:
            committee.load()

        new_datapoints = committee.select_data(prop=self.prop, red=self.red)
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
                    '\npython ../../cp2k_main.py {} {} {} {}'.format(self.cp2k_walltime, self.cp2k_cores, str(self.input_dir), str(self.output_dir), self.cp2k_cores, 'restart.yaml')
                )
            config = Config()
            config.cycle = self.cycle
            config.walltime = self.walltime
            config.cores = self.cores
            config.traj_index = self.traj_index
            config.env = self.env
            config.cluster = self.cluster
            config.save(str(restart_conf), 'yaml')

            shell = VSC_shell(ssh_keys.HOST, ssh_keys.USERNAME, ssh_keys.PASSWORD, ssh_keys.KEY_FILENAME)
            shell.submit_job(self.cp2k_cluster, cp2k_dir.resolve(), 'job.sh')
            shell.__del__()
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
            elapsed_t = time.time()-self.start_time
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
                    logging.info('\nTotal elapsed time: {} hours of total {} hours'.format(round((time.time()-self.start_time)/3600,3),round(self.walltime.seconds/3600,3)))
                    
    def start_next_cyle(self):
        old = self.traj_index.split(':')
        old_len = int(old[1]) - int(old[0])
        index = '{}:{}'.format(int(old[0])+old_len, int(old[1])+old_len)        
        next_walltime = self.walltime + self.len_models*self.walltime_per_model_add
        if self.cp2k:
            first_walltime = self.cp2k_qbc_walltime
        else:
            first_walltime = next_walltime

        if int(old[1])+old_len <= self.max_index:
            with open('cycle{}.sh'.format(self.cycle+1),'w') as rsh:
                rsh.write(
                    '#!/bin/sh'
                    '\n\n#PBS -o cycle{}_o.txt'
                    '\n#PBS -e cycle{}_e.txt'
                    '\n#PBS -l walltime={}'
                    '\n#PBS -l nodes=1:ppn={}:gpus=1'
                    '\n\nsource ~/.{}'
                    '\npython ../train.py {} {} --traj-index {}'.format(self.cycle+1, self.cycle+1, str(first_walltime), self.cores, self.env, self.cycle+1, str(next_walltime), index)
                )
            os.system('module swap cluster/{}; qsub cycle{}.sh -d $(pwd)'.format(self.cluster, self.cycle+1))
