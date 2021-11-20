import time
start = time.time()
from pathlib import Path
import logging
import shutil
import os 

import ase.io
from nequip.utils import Config

from qbc import qbc
logging.basicConfig(format='',level=logging.INFO)

##########################################################################################

head_dir = Path('/scratch/gent/vo/000/gvo00003/vsc43785/Thesis/query/committee_train')
traj_dir = head_dir / 'unknown.xyz'
cycle = 1
n_select = 10
dataset_len = 1050
n_val = 50
max_epochs = 203
walltime = '02'

##########################################################################################
logging.info('___ QUERY BY COMMITTEE ___\n')
logging.info('\t\t###########')
logging.info('\t\t# CYCLE {} #'.format(cycle))
logging.info('\t\t###########')

assert traj_dir.exists(), 'Trajectory path does not exist'
prev_name = 'cycle{}'.format(cycle-1)
prev_nequip_train_dir = head_dir / prev_name / 'results'
prev_dataset_dir = head_dir / prev_name / 'data.xyz'
assert prev_nequip_train_dir.is_dir(), 'Previous training directory does not exist'
assert prev_dataset_dir.exists(), 'Previous training dataset directory does not exist'

name = 'cycle{}'.format(cycle)
cycle_dir = head_dir / name
dataset_dir = cycle_dir / 'data.xyz'

def evaluate_committee():
    committee = qbc(name=name, models_dir=prev_nequip_train_dir, 
                    traj_dir=traj_dir, results_dir=head_dir,
                    traj_index=':100', n_select=n_select, nequip_train=True
                )
    assert cycle_dir.is_dir(), 'OOPS'

    committee.evaluate_committee(save=True)
    #committee.load()
    committee.plot_traj_disagreement()

    new_datapoints = committee.select_data('forces','mean')
    prev_dataset = ase.io.read(prev_dataset_dir,format='extxyz',index=':')
    new_dataset = new_datapoints + prev_dataset

    ase.io.write(dataset_dir,new_dataset,format='extxyz')
    logging.info('Saved the new dataset of length {}'.format(len(new_dataset)))

#evaluate_committee()
assert dataset_dir.exists(), 'OOPS'
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

    config.n_train = dataset_len + n_select - n_val
    config.n_val = n_val

    config.max_epochs = max_epochs

    return config

p = Path(nequip_train_dir).glob('**/*')
model_files = [x for x in p if x.is_dir()]

#elapsed_t = time.time()-start
#left_t = walltime*3600 - elapsed_t
#logging.info('\n## {} hours left over of HPC walltime'.format(round(left_t/3600,3)))
#t_per_model = left_t/len_models
#logging.info('## Each of the {} models will train {} hours'.format(len_models,round(t_per_model/3600,3)))
hpc_dir = cycle_dir / 'hpc'
if not hpc_dir.exists():
    hpc_dir.mkdir()

os.system('module swap cluster/joltik')

for file in sorted(model_files):
    logging.info('\n############################################\n')
    logging.info('Starting retraining of {}\n'.format(file.name))
    config = make_config()
    train_dir = file / 'trainer.pth'
    #restart(train_dir, config)

    train_dir=str(train_dir)

    hpc_run_dir = hpc_dir / file.name
    if not hpc_run_dir.exists():
        hpc_run_dir.mkdir()

    config.save(str(hpc_run_dir / 'updated_config.yaml'),'yaml')
    config_dir = str(hpc_run_dir / 'updated_config.yaml')
    
    with open(hpc_run_dir / 'run.sh','w') as rsh:
        rsh.write(
    '''\
    #!/bin/sh

    #PBS -l walltime={}:00:00
    #PBS -l nodes=1:ppn=8:gpus=1

    source ~/.torchenv

    python restart.py {} --update-config {} 
    '''.format(walltime,train_dir,config_dir)
    )

    os.system('qsub {} -d $(pwd) -e {} -o {}'.format(hpc_run_dir / 'run.sh', hpc_run_dir / 'error', hpc_run_dir / 'output'))

    break

