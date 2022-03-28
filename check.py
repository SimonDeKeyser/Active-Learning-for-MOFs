from pathlib import Path
import argparse
import os
from nequip.utils import Config
import pandas as pd
import ase.io

parser = argparse.ArgumentParser(
            description="Check if all jobs are finished, if not, restart them in a smart way."
        )
parser.add_argument("cycle", help="The number of the QbC cycle ")
parser.add_argument("function", help="The function to check")
args = parser.parse_args()

def check_MD(cycle):
    config = Config.from_file(str(Path.cwd() / '../qbc_train' / 'cycle{}'.format(cycle) / 'params.yaml'), 'yaml')
    md_dir = Path.cwd() / '../qbc_train' / 'cycle{}'.format(cycle) / 'MD'
    p = md_dir.glob('*')
    files = [x for x in p]
    for f in files:
        if not (f / 'new_data.xyz').is_file():
            print('{} MD not finished, restarting ...'.format(f.name))
            os.system('module swap cluster/{}; cd {} ;qsub job.sh -d $(pwd)'.format(config.cp2k_cluster, f))

def check_CP2K(cycle):
    config = Config.from_file(str(Path.cwd() / '../qbc_train' / 'cycle{}'.format(cycle) / 'params.yaml'), 'yaml')
    md_dir = Path.cwd() / '../qbc_train' / 'cycle{}'.format(cycle) / 'MD'
    p = md_dir.glob('*')
    files = [x for x in p]
    for f in files:
        if not (f / 'finished').is_file():
            print('{} CP2K not finished, restarting ...'.format(f.name))
            os.system('module swap cluster/{}; cd {}/cp2k ;qsub job.sh -d $(pwd)'.format(config.cp2k_cluster, f))

def log(cycle):
    config = Config.from_file(str(Path.cwd() / '../qbc_train' / 'cycle{}'.format(cycle) / 'params.yaml'), 'yaml')
    os.system('module swap cluster/{}; qstat'.format(config.cp2k_cluster))
    md_dir = Path.cwd() / '../qbc_train' / 'cycle{}'.format(cycle) / 'MD'
    p = md_dir.glob('*')
    files = [x for x in p]
    df = pd.DataFrame(index=sorted([x.name for x in files]), columns=['MD', 'CP2K'])
    
    for f in files:
        if (f / 'job.sh').is_file():
            df['MD'][f.name] = 'submitted'
        else:
            df['MD'][f.name] = '...'

        if (f / 'trajs').is_dir():
            p = (f / 'trajs').glob('*.traj')
            current_traj = max([int(x.name[:-5]) for x in p])
            df['MD'][f.name] = 'traj {}'.format(current_traj)
        
        if (f / 'qbc').is_dir():
            df['MD'][f.name] = 'QbC'

        if (f / 'new_data.xyz').is_file():
            df['MD'][f.name] = 'done'

        if (f / 'cp2k' / 'job.sh').is_file():
            df['CP2K'][f.name] = 'submitted'
        else:
            df['CP2K'][f.name] = '...'

        if (f / 'calculated_data.xyz').is_file():  
            inp = ase.io.read(f / 'new_data.xyz', index=':')
            outp = ase.io.read(f / 'calculated_data.xyz', index=':')
            df['CP2K'][f.name] = '{}/{}'.format(len(outp), len(inp))
    print('MD TRAJECTORIES:')
    print(df)
    print('\n')
    #df.to_csv('/runs/cycle{}.csv'.format(cycle), sep='\t')

    hpc_dir = Path.cwd() / '../qbc_train' / 'cycle{}'.format(cycle) / 'hpc'
    if hpc_dir.is_dir():
        os.system('module swap cluster/accelgor; qstat')
        results_dir = hpc_dir.parents[0] / 'results'
        p = hpc_dir.glob('*')
        files = [x for x in p]
        df = pd.DataFrame(columns=sorted([x.name for x in files]), index=['epoch', 'walltime [h]', 'status'])
        for f in files:
            if (results_dir / f.name / 'metrics_epoch.csv').is_file():
                if  (f / 'error.txt').is_file():   
                    df[f.name]['status'] = 'training'
                    metrics = pd.read_csv(results_dir / f.name / 'metrics_epoch.csv')
                    df[f.name]['epoch'] = metrics['epoch'].values[-1]
                    df[f.name]['walltime [h]'] = metrics[' wall'].values[-1]/3600
                else:
                    df[f.name]['status'] = 'submitted'
                    df[f.name]['epoch'] = '...'
                    df[f.name]['walltime [h]'] = '...'
                
                if (f / 'finished').is_file():
                    df[f.name]['status'] = 'done'
          
        print('TRAINING:')
        print('Total walltime: {}'.format(config.walltime))
        print(df)

if args.function == 'MD':
    print('Make sure that the jobs are not still running with the log command\n')
    x = input('Restart MD runs that have not yet finished? (y/n): ')
    if x == 'y':
        check_MD(int(args.cycle))
elif args.function == 'CP2K':
    print('Make sure that the jobs are not still running with the log command\n')
    x = input('Restart CP2K runs that have not yet finished? (y/n): ')
    if x == 'y':
        check_CP2K(int(args.cycle))
elif args.function == 'log':
    log(int(args.cycle))
else:
    print('No valid command')