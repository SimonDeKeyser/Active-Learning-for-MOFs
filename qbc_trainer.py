from pathlib import Path
import logging
import shutil

import ase.io
from nequip.scripts.restart import restart
from nequip.utils import Config

from qbc import qbc
logging.basicConfig(format='',level=logging.INFO)
##########################################################################################

head_dir = Path('/scratch/gent/vo/000/gvo00003/vsc43785/Thesis/query/committee_train')
traj_dir = head_dir / 'unknown.xyz'
cycle = 1
n_select = 2
dataset_len = 1050
n_val = 50
max_epochs = 2

##########################################################################################
logging.info('### Query by Commitee ###\n')
logging.info('Cycle {}:'.format(cycle))

assert traj_dir.exists(), 'Trajectory path does not exist'
prev_name = 'cycle{}'.format(cycle-1)
prev_nequip_train_dir = head_dir / prev_name / 'results'
prev_dataset_dir = head_dir / prev_name / 'data.xyz'
assert prev_nequip_train_dir.is_dir(), 'Previous training directory does not exist'
assert prev_dataset_dir.exists(), 'Previous training dataset directory does not exist'

name = 'cycle{}'.format(cycle)
cycle_dir = head_dir / name
dataset_dir = cycle_dir / 'data.xyz'

def evaluate_commitee():
    committee = qbc(name=name, models_dir=prev_nequip_train_dir, 
                    traj_dir=traj_dir, results_dir=head_dir,
                    traj_index=':10', n_select=n_select, nequip_train=False
                )
    assert cycle_dir.is_dir(), 'OOPS'

    #committee.evaluate_committee(save=True)
    committee.load()
    committee.plot_traj_disagreement()

    new_datapoints = committee.select_data('forces','mean')
    prev_dataset = ase.io.read(prev_dataset_dir,format='extxyz',index=':')
    new_dataset = new_datapoints + prev_dataset

    ase.io.write(dataset_dir,new_dataset,format='extxyz')
    logging.info('Saved the new dataset of length {}'.format(len(new_dataset)))

assert dataset_dir.exists(), 'OOPS'
nequip_train_dir = cycle_dir / 'results'
if not nequip_train_dir.exists():
    nequip_train_dir.mkdir()

p = Path(prev_nequip_train_dir).glob('**/*')
model_files = [x for x in p if x.is_dir()]

for file in sorted(model_files):
    if not file.name == 'processed':
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

    config.save(str(cycle_dir / 'conf.yaml'),'yaml')
    return config

config = make_config()
test_file = nequip_train_dir / 'model0' / 'trainer.pth'
restart(test_file, config)