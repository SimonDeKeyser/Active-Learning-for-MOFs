from pathlib import Path
import logging
import os
import pandas as pd
from natsort import natsorted

from nequip.utils import Config
import torch

logging.basicConfig(format='',level=logging.INFO)
##########################################################################################

do_first = False
root = Path('../').resolve() 
head_dir = root / 'qbc_train'
test_dir = head_dir / 'trajectory.xyz'      
index = '30000:34606'   
walltime = '00:05:00'
first_walltime = '00:05:00'
batch_size = 5
device = 'cuda'    
eval_name = 'evaluation'                                                                                                            

##########################################################################################
logging.info('EVALUATION ON TEST SET:\n')

assert head_dir.exists(), 'Head directory does not exist'
assert test_dir.exists(), 'Test path does not exist'
logging.info(str(test_dir))
logging.info('Indices: [{}]'.format(index))

p = head_dir.glob('*')
cycle_names = natsorted([x.name for x in p if (x.is_dir() and x.name[:5] == 'cycle')])
cycles = [(head_dir / name) for name in cycle_names]

evaluate_dir = head_dir / eval_name
if not evaluate_dir.exists():
    evaluate_dir.mkdir()

index_list = index.split(':')
indices = torch.arange(int(index_list[0]), int(index_list[1]))
test_index = evaluate_dir / 'test_indexes.pt'
torch.save(indices, test_index)

def make_dataset_config(train_config, hpc_run_dir):
    config = Config()

    config.root = str(evaluate_dir)
    config.r_max = train_config['r_max']

    config.dataset = 'ase'
    config.dataset_file_name = str(test_dir)
    config.save(str(hpc_run_dir / 'dataset_config.yaml'),'yaml')

    return str(hpc_run_dir / 'dataset_config.yaml')

def make_metrics_config(train_config, hpc_run_dir):
    config = Config()

    config.metrics_components = train_config['metrics_components']    
    config.save(str(hpc_run_dir / 'metrics_config.yaml'),'yaml')

    return str(hpc_run_dir / 'metrics_config.yaml')

def evaluate(nequip_train_dir):
    p = Path(nequip_train_dir).glob('*')
    model_files = [x for x in p if x.is_dir()]

    for file in sorted(model_files):
        if (not file.name == 'processed') and (not file == cycles[0] / 'results' / 'model0'):
            logging.info('\n###################################################################################################\n')
            logging.info('Starting evaluation of:\n')
            logging.info(file)
            config = Config.from_file(str(file / 'config_final.yaml'))

            hpc_run_dir = hpc_dir / file.name
            if not hpc_run_dir.exists():
                hpc_run_dir.mkdir()

            dataset_config = make_dataset_config(config, hpc_run_dir)
            metrics_config = make_metrics_config(config, hpc_run_dir)

            with open(hpc_run_dir / 'eval.sh','w') as rsh:
                rsh.write(
                    '#!/bin/sh'
                    '\n\n#PBS -l walltime={}'
                    '\n#PBS -l nodes=1:ppn=8:gpus=1'
                    '\n\nsource ~/.torchenv'
                    '\nnequip-evaluate --train-dir {} --dataset-config {} '
                    '--metrics-config {} --batch-size {} --test-indexes {} '
                    '--device {}'.format(walltime, str(file), dataset_config, metrics_config, batch_size, test_index, device)
                )
            
            os.system('module swap cluster/joltik; qsub {} -d $(pwd) -e {} -o {}'.format(hpc_run_dir / 'eval.sh', hpc_run_dir / 'error', hpc_run_dir / 'output'))
            logging.info('\t-{} submitted'.format(file.name))

def evaluate_first(cycle = cycles[0]):
    nequip_train_dir = cycle / 'results'
    assert nequip_train_dir.exists(), 'The NequIP training directory cannot be found'
    file = nequip_train_dir / 'model0'
    logging.info('\n###################################################################################################\n')
    logging.info('Starting evaluation of:\n')
    logging.info(file)
    config = Config.from_file(str(file / 'config_final.yaml'))

    hpc_dir = evaluate_dir / cycle.name
    if not hpc_dir.exists():
        hpc_dir.mkdir()

    hpc_run_dir = hpc_dir / file.name
    if not hpc_run_dir.exists():
        hpc_run_dir.mkdir()

    dataset_config = make_dataset_config(config, hpc_run_dir)
    metrics_config = make_metrics_config(config, hpc_run_dir)

    with open(hpc_run_dir / 'eval.sh','w') as rsh:
        rsh.write(
            '#!/bin/sh'
            '\n\n#PBS -l walltime={}'
            '\n#PBS -l nodes=1:ppn=8:gpus=1'
            '\n\nsource ~/.torchenv'
            '\nnequip-evaluate --train-dir {} --dataset-config {} '
            '--metrics-config {} --batch-size {} --test-indexes {} '
            '--device {}'.format(first_walltime, str(file), dataset_config, metrics_config, batch_size, test_index, device)
        )
    
    os.system('module swap cluster/joltik; qsub {} -d $(pwd) -e {} -o {}'.format(hpc_run_dir / 'eval.sh', hpc_run_dir / 'error', hpc_run_dir / 'output'))
    logging.info('\t-{} submitted'.format(file.name))

if do_first:
    evaluate_first()
else:
    for cycle in cycles:
        nequip_train_dir = cycle / 'results'
        assert nequip_train_dir.exists(), 'The NequIP training directory cannot be found'

        hpc_dir = evaluate_dir / cycle.name
        if not hpc_dir.exists():
            hpc_dir.mkdir()

        evaluate(nequip_train_dir)
