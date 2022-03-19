from time import perf_counter
from dataclasses import dataclass

import sys
if sys.version_info[1] >= 8:
    from typing import Final
else:
    from typing_extensions import Final

from pathlib import Path, PosixPath
import logging
import shutil
import os 
import random
import datetime as dt
from datetime import timedelta

import ase.io
from nequip.utils import Config
from nequip.scripts.train import _set_global_options
from nequip.train import Trainer
from nequip.utils import Config
from nequip.utils.versions import check_code_version, get_config_code_versions
import pandas as pd
import numpy as np
import torch
from e3nn.util.jit import script

CONFIG_KEY: Final[str] = "config"
NEQUIP_VERSION_KEY: Final[str] = "nequip_version"
TORCH_VERSION_KEY: Final[str] = "torch_version"
E3NN_VERSION_KEY: Final[str] = "e3nn_version"
CODE_COMMITS_KEY: Final[str] = "code_commits"
R_MAX_KEY: Final[str] = "r_max"
N_SPECIES_KEY: Final[str] = "n_species"
TYPE_NAMES_KEY: Final[str] = "type_names"
JIT_BAILOUT_KEY: Final[str] = "_jit_bailout_depth"
TF32_KEY: Final[str] = "allow_tf32"

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
    root: PosixPath = Path('../../').resolve()
    head_dir: PosixPath = root / 'qbc_train'
    traj_dir: PosixPath = head_dir / 'trajectory.xyz'                                                                                                                             
    n_select: int = 11
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
    cores: int = 12
    cp2k_cores: int = 24
    cp2k_walltime: str = '01:00:00'
    traj_index: str = ':'
    cp2k_cluster: str = 'doduo'

    def __post_init__(self):
        logging.info('___ QUERY BY COMMITTEE ___\n')
        logging.info('\t\t###########')
        logging.info('\t\t# CYCLE {} #'.format(self.cycle))
        logging.info('\t\t###########\n')

        self.start_time = None
        assert self.head_dir.exists(), 'Head directory does not exist'
        assert self.traj_dir.exists(), 'Trajectory path does not exist'
        prev_name = 'cycle{}'.format(self.cycle-1)
        self.prev_nequip_train_dir = self.head_dir / prev_name / 'results'
        self.prev_dataset_dir = self.head_dir / prev_name / 'data.xyz'
        assert self.prev_nequip_train_dir.is_dir(), 'Previous training directory does not exist'
        assert self.prev_dataset_dir.exists(), 'Previous training dataset directory does not exist'

        self.name = 'cycle{}'.format(self.cycle)
        self.cycle_dir = self.head_dir / self.name
        if not self.cycle_dir.exists():
            self.cycle_dir.mkdir()
            logging.info('Created cycle directory: \n{}'.format(self.cycle_dir))
        self.dataset_dir = self.cycle_dir / 'data.xyz'
        self.input_dir = self.cycle_dir / 'new_data.xyz'
        self.output_dir = self.cycle_dir / 'calculated_data.xyz'

    def deploy(self, train_dir):
        config = Config.from_file(str(train_dir / "config.yaml"))
        _set_global_options(config)
        check_code_version(config)
        model, _ = Trainer.load_model_from_training_session(
            train_dir, model_name="best_model.pth", device="cpu"
        )

        model.eval()
        if not isinstance(model, torch.jit.ScriptModule):
            model = script(model)
        logging.info("Compiled & optimized model.")

        metadata: dict = {}
        code_versions, code_commits = get_config_code_versions(config)
        for code, version in code_versions.items():
            metadata[code + "_version"] = version
        if len(code_commits) > 0:
            metadata[CODE_COMMITS_KEY] = ";".join(
                f"{k}={v}" for k, v in code_commits.items()
            )

        metadata[R_MAX_KEY] = str(float(config["r_max"]))
        if "allowed_species" in config:
            # This is from before the atomic number updates
            n_species = len(config["allowed_species"])
            type_names = {
                type: ase.data.chemical_symbols[atomic_num]
                for type, atomic_num in enumerate(config["allowed_species"])
            }
        else:
            # The new atomic number setup
            n_species = str(config["num_types"])
            type_names = config["type_names"]
        metadata[N_SPECIES_KEY] = str(n_species)
        metadata[TYPE_NAMES_KEY] = " ".join(type_names)

        metadata[JIT_BAILOUT_KEY] = str(config["_jit_bailout_depth"])
        metadata[TF32_KEY] = str(int(config["allow_tf32"]))
        metadata[CONFIG_KEY] = (train_dir / "config.yaml").read_text()

        metadata = {k: v.encode("ascii") for k, v in metadata.items()}
        torch.jit.save(model, train_dir / 'deployed.pth', _extra_files=metadata)


    def create_trajectories(self):
        config = Config()
        config.cycle = self.cycle
        config.walltime = self.walltime
        config.cores = self.cores
        config.traj_index = self.traj_index
        config.env = self.env
        config.cluster = self.cluster
        config.walltime_per_model_add = self.walltime_per_model_add
        config.cp2k_walltime = self.cp2k_walltime
        config.cp2k_cores = self.cp2k_cores
        config.cp2k_cluster = self.cp2k_cluster
        config.save(str(self.cycle_dir / 'params.yaml'), 'yaml')

        p = (self.prev_nequip_train_dir).glob('*')
        model_dirs = [x for x in p if (x.is_dir() and x.name[:5] == 'model')]

        best_mae = 100
        best_model = None
        logging.info('Deploying all models and finding best NNP model for MD:')
        for model_dir in sorted(model_dirs):
            logging.info(model_dir.name)
            metrics = pd.read_csv(model_dir / 'metrics_epoch.csv')
            clean_df = metrics[metrics[' wall'] != ' wall']
            mae = clean_df[' validation_all_f_mae'].to_numpy(dtype=np.float64).min()
            logging.info('{} eV/A'.format(mae))
            if mae < best_mae:
                best_mae = mae
                best_model = model_dir
            self.deploy(model_dir)

        logging.info('Best model for MD simulations: {}'.format(best_model.name))
        
        p = Path(self.traj_dir).glob('**/*.xyz')
        traj_files = [x for x in p]

        md_dir = self.cycle_dir / 'MD'
        if not md_dir.exists():
            md_dir.mkdir()

        logging.info('\nStarting MD job:')
        shell = VSC_shell(ssh_keys.HOST, ssh_keys.USERNAME, ssh_keys.PASSWORD, ssh_keys.KEY_FILENAME)
        for f in sorted(traj_files):
            logging.info('\t - {}'.format(f.name))
            job_dir = md_dir / f.name[:-4]
            if not job_dir.exists():
                job_dir.mkdir()
            with open(job_dir / 'job.sh' ,'w') as rsh:
                rsh.write(
                    '#!/bin/sh'
                    '\n\n#PBS -o output.txt'
                    '\n#PBS -e error.txt'
                    '\n#PBS -l walltime=01:30:00'
                    '\n#PBS -l nodes=1:ppn=12'
                    '\n\nsource ~/.cp2kenv'
                    '\npython ../../../../QbC/md.py {} {} {}'.format(best_model / 'deployed.pth', f, self.n_select)
                )
            shell.submit_job(self.cp2k_cluster, job_dir.resolve(), 'job.sh')

        shell.__del__()

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
        committee = QbC(results_dir=self.cycle_dir, cnnp=cnnp, traj_dir=self.traj_dir,
                        traj_index=self.traj_index, n_select=self.n_select
                        )

        if not self.load_query_results:
            committee.evaluate_committee(save=True)
        else:
            committee.load()

        new_datapoints = committee.select_data(prop=self.prop, red=self.red)
        committee.plot_traj_disagreement()
 
        ase.io.write(self.output_dir, new_datapoints, format='extxyz')

    def make_config(self, prev_n_train, prev_n_val):
        config = Config()

        config.root = str(self.nequip_train_dir)
        config.restart = True
        config.append = True

        config.dataset_file_name = str(self.dataset_dir)

        config.wandb_resume = True

        config.n_train = prev_n_train + self.n_train_added
        config.n_val = prev_n_val + self.n_val_added

        config.max_epochs = self.max_epochs
        config.lr_scheduler_name = 'none'

        return config
    
    def run_hpc_job(self, hpc_run_dir, train_dir, config_dir):
        with open(hpc_run_dir / 'job.sh','w') as rsh:
            rsh.write(
                '#!/bin/sh'
                '\n\n#PBS -o output.txt'
                '\n#PBS -e error.txt'
                '\n#PBS -l walltime={}'
                '\n#PBS -l nodes=1:ppn={}:gpus=1'
                '\n\nsource ~/.{}'
                '\npython ../../../../QbC/restart.py {} --update-config {} --walltime {}'.format(str(self.walltime), self.cores, self.env, train_dir, config_dir, str(self.walltime))
            )

        os.system('module swap cluster/{}; cd {}; qsub job.sh -d $(pwd)'.format(self.cluster, hpc_run_dir))

    def dataset_from_md(self):
        prev_dataset = ase.io.read(self.prev_dataset_dir, format='extxyz',index=':')
        md_dir = self.cycle_dir / 'MD'
        p = md_dir.glob('**/*/calculated_data.xyz')
        len_train_add = self.n_select - self.n_val_add
        logging.info('Creating new dataset from MD trajectories:')
        train_add = []
        val_add = []
        for file in p:
            logging.info('*\t {}'.format(file.parts[-2]))
            new_datapoints = ase.io.read(file, index = ':', format='extxyz')
            random.shuffle(new_datapoints)
            train_add += new_datapoints[:len_train_add]
            val_add += new_datapoints[len_train_add:]
        new_dataset = train_add + prev_dataset + val_add

        self.n_train_added = len(train_add)
        self.n_val_added = len(val_add)
        ase.io.write(self.dataset_dir, new_dataset, format='extxyz')
        logging.info('Saved the new dataset of length {}'.format(len(new_dataset)))

    def dataset_from_traj(self):
        new_datapoints = ase.io.read(self.output_dir, index = ':', format='extxyz')
        prev_dataset = ase.io.read(self.prev_dataset_dir, format='extxyz',index=':')
        assert self.n_val_add < self.n_select, 'No training points added but only validation points'
        len_train_add = self.n_select - self.n_val_add
        random.shuffle(new_datapoints)
        new_dataset = new_datapoints[:len_train_add] + prev_dataset + new_datapoints[len_train_add:]

        self.n_train_added = len(len_train_add)
        self.n_val_added = len(new_datapoints) - len(len_train_add)
        ase.io.write(self.dataset_dir, new_dataset, format='extxyz')
        logging.info('Saved the new dataset of length {}'.format(len(new_dataset)))

    def restart_training(self):
        self.nequip_train_dir = self.cycle_dir / 'results'
        if not self.nequip_train_dir.exists():
            self.nequip_train_dir.mkdir()

        p = Path(self.prev_nequip_train_dir).glob('model*')
        model_files = [x for x in p if x.is_dir()]

        self.len_models = 0
        for file in sorted(model_files):
            self.len_models += 1
            if not (self.nequip_train_dir / file.name).exists():
                shutil.copytree(file, self.nequip_train_dir / file.name)
                logging.info('Copying NequIP train folder of {}'.format(file.name))

        hpc_dir = self.cycle_dir / 'hpc'
        if not hpc_dir.exists():
            hpc_dir.mkdir()

        if not self.send_hpc_run:
            elapsed_t = perf_counter()-self.start_time
            left_t = self.walltime.seconds - elapsed_t
            logging.info('\n## {} hours left over of HPC walltime'.format(round(left_t/3600,3)))
            t_per_model = left_t/self.len_models
            logging.info('## Each of the {} models will train {} hours'.format(self.len_models,round(t_per_model/3600,3)))

        p = Path(self.nequip_train_dir).glob('model*')
        model_files = [x for x in p if x.is_dir()]

        for file in sorted(model_files):
            if 'model' in file.name:
                logging.info('\n###################################################################################################\n')
                logging.info('Starting retraining of {}\n'.format(file.name))
                prev_config = Config.from_file(str(self.prev_nequip_train_dir/ file.name / 'config.yaml'), 'yaml')
                prev_n_train = prev_config.n_train
                prev_n_val = prev_config.n_val

                config = self.make_config(prev_n_train, prev_n_val)

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
                    logging.info('\nTotal elapsed time: {} hours of total {} hours'.format(round((perf_counter()-self.start_time)/3600,3),round(self.walltime.seconds/3600,3)))
                    
    def start_next_cyle(self, md=False):
        next_walltime = self.walltime + self.len_models*self.walltime_per_model_add
        if md:
            with open('cycle{}.sh'.format(self.cycle+1),'w') as rsh:
                rsh.write(
                    '#!/bin/sh'
                    '\n\n#PBS -o cycle{}_o.txt'
                    '\n#PBS -e cycle{}_e.txt'
                    '\n#PBS -l walltime={}'
                    '\n#PBS -l nodes=1:ppn={}:gpus=1'
                    '\n\nsource ~/.{}'
                    '\npython ../train.py {} {}'.format(self.cycle+1, self.cycle+1, str(next_walltime), self.cores, self.env, self.cycle+1, str(next_walltime))
                )
            os.system('module swap cluster/{}; bash cycle{}.sh'.format(self.cluster, self.cycle+1))

        else:
            old = self.traj_index.split(':')
            old_len = int(old[1]) - int(old[0])
            index = '{}:{}'.format(int(old[0])+old_len, int(old[1])+old_len)        

            if int(old[1])+old_len <= self.max_index:
                with open('cycle{}.sh'.format(self.cycle+1),'w') as rsh:
                    rsh.write(
                        '#!/bin/sh'
                        '\n\n#PBS -o cycle{}_o.txt'
                        '\n#PBS -e cycle{}_e.txt'
                        '\n#PBS -l walltime={}'
                        '\n#PBS -l nodes=1:ppn={}:gpus=1'
                        '\n\nsource ~/.{}'
                        '\npython ../train.py {} {} --traj-index {}'.format(self.cycle+1, self.cycle+1, str(next_walltime), self.cores, self.env, self.cycle+1, str(next_walltime), index)
                    )
                os.system('module swap cluster/{}; qsub cycle{}.sh -d $(pwd)'.format(self.cluster, self.cycle+1))
