from pathlib import Path
import logging
import os
import pandas as pd

from nequip.utils import Config
import torch

logging.basicConfig(format='',level=logging.INFO)
##########################################################################################

root = Path('../').resolve() 
head_dir = root / 'qbc_train'
cycle_dir = head_dir / 'cycle6'
test_dir = head_dir / 'trajectory.xyz'      
index = '30000:34606'   
walltime = '00:30:00'
batch_size = 5
device = 'cuda'  
log = 'None'                                                                                                                 

##########################################################################################
logging.info('EVALUATION ON TEST SET:\n')

assert head_dir.exists(), 'Head directory does not exist'
assert test_dir.exists(), 'Test path does not exist'
logging.info(str(test_dir))
logging.info('Indices: [{}]'.format(index))

nequip_train_dir = cycle_dir / 'results'
assert nequip_train_dir.exists(), 'The NequIP training directory cannot be found'

evaluate_dir = head_dir / 'evaluation'
if not evaluate_dir.exists():
    evaluate_dir.mkdir()

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

hpc_dir = evaluate_dir / 'hpc'
if not hpc_dir.exists():
    hpc_dir.mkdir()

def evaluate():
    index_list = index.split(':')
    indices = torch.arange(int(index_list[0]), int(index_list[1]))
    test_index = evaluate_dir / 'test_indexes.pt'
    torch.save(indices, test_index)

    p = Path(cycle_dir / nequip_train_dir).glob('**/*')
    model_files = [x for x in p if x.is_dir()]

    for file in sorted(model_files):
        if not file.name == 'processed':
            logging.info('\n###################################################################################################\n')
            logging.info('Starting evaluation of {}\n'.format(file.name))
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
            
            break

evaluate()

def results():
    p = Path(hpc_dir).glob('**/*')
    model_files = [x for x in p if x.is_dir()]
    data = {}
    model_names = []
    f_mae = []
    e_mae = []
    for file in sorted(model_files):
        model_names.append(file.name)
        f = open(file / 'output', 'r')
        lines = f.readlines()
        for line in lines:
            if '=' in line:
                metric, value = line.split('=')
                metric = metric.strip()
                if metric == 'all_f_mae':
                    f_mae.append(1000*float(value))
                if metric == 'e_mae':
                    e_mae.append(1000*float(value))
    data['F MAE [meV/$\AA$]'] = f_mae
    data['E MAE [meV/$\AA$]'] = e_mae
    df = pd.DataFrame.from_dict(data)
    df.index = model_names
    print('\nResults:\n')
    print(df)

#results()