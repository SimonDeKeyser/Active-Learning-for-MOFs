from pathlib import Path
import logging
import os

from nequip.utils import Config

logging.basicConfig(format='',level=logging.INFO)

'''
IMPORTANT:
    Before making the QbC environment and doing the first training,
    the folder (called QbC) containing this python script should be in a certain root directory
    and that root directory should contain:
        * A qbc training folder, ex.: qbc_train
        * A folder called confs, that contains a NequIP config file called config0.yaml
    
    The qbc training folder should contain:
        * A folder called cycle0
        * And cycle0 contains the first dataset called data.xyz

Parameters:
    - head_dir                  Head directory for the QbC environment     
    - len_models                The amount of models you want in the committee
    - wandb_project             The name of the wandb project 
    - n_train                   The number of datapoints in the first training set
    - n_val                     The number of datapoints in the first validation set
    - walltime                  The walltime for the first training
'''
##########################################################################################

root = Path('../').resolve()
head_dir = root / 'qbc_train'
len_models = 4
wandb_project = 'q4_random10'
n_train = 10
n_val = 1
walltime = '00:10:00'

##########################################################################################

def check_init_state():
    logging.info('#Checking the contents of the qbc folders ...\n')
    if (head_dir / 'cycle0' / 'data.xyz').exists():
        cycle_dir = head_dir / 'cycle0'
        ok = True
    else:
        ok = False
        logging.info('data.xyz not in cycle0 folder of head directory')
    if (root / 'confs' / 'config0.yaml').exists():
        conf_dir = root / 'confs'
        ok = True
    else:
        ok = False
        logging.info('config0.yaml not in confs folder of root directory')
    assert ok, 'Check the necessary directory setup for initializing QbC'
    logging.info('\t-OK!')
    return cycle_dir, conf_dir

def make_configs():
    logging.info('#Making the config files ...\n')
    for i in range(len_models):
        file = 'config{}.yaml'.format(i)
        if not (conf_dir / file).exists():
            config = Config.from_file(str(conf_dir / 'config{}.yaml'.format(i-1)))
        else:
            config = Config.from_file(str(conf_dir / 'config{}.yaml'.format(i)))

        config.root = str(cycle_dir / 'results')
        config.run_name = 'model{}'.format(i)
        config.seed = i

        config.dataset_file_name = str(cycle_dir / 'data.xyz')
        config.wandb_project = wandb_project

        config.n_train = n_train
        config.n_val = n_val

        config.train_val_split = 'sequential'

        config.save(str(conf_dir / 'config{}.yaml'.format(i)),'yaml')
    logging.info('\t-Done!')

def make_train_scripts():
    logging.info('#Making the training scripts ...\n')
    if not (cycle_dir / 'hpc').exists():
        (cycle_dir / 'hpc').mkdir()
    hpc_dir = cycle_dir / 'hpc'

    for i in range(len_models):
        file = 'model{}'.format(i)
        if not (hpc_dir / file).exists():
            (hpc_dir / file).mkdir()

        with open(hpc_dir / file / 'train.sh','w') as rsh:
            rsh.write(
                '#!/bin/sh'
                '\n\n#PBS -l walltime={}'
                '\n#PBS -l nodes=1:ppn=8:gpus=1'
                '\n\nsource ~/.torchenv'
                '\nnequip-train {}'.format(walltime,str(conf_dir / 'config{}.yaml'.format(i)))
            )
    logging.info('\t-Done!')
    return hpc_dir

def start_training():
    logging.info('#Starting the training ...\n')
    for i in range(len_models):
        file = hpc_dir / 'model{}'.format(i)
        os.system('module swap cluster/joltik; qsub {} -d $(pwd) -e {} -o {}'.format(file / 'train.sh', file / 'error', file / 'output'))
        logging.info('\t-model{} submitted'.format(i))

cycle_dir, conf_dir = check_init_state()

make_configs()

hpc_dir = make_train_scripts()

start_training()
