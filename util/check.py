from pathlib import Path
import argparse
import os
from nequip.utils import Config
import pandas as pd
import ase.io
import numpy as np

import logging 
logging.basicConfig(format='',level=logging.INFO)

parser = argparse.ArgumentParser(
            description="Check if all jobs are finished, if not, restart them in a smart way."
        )
parser.add_argument("cycle", help="The number of the QbC cycle ")
parser.add_argument("function", help="The function to check")
parser.add_argument("--restart", help="Restart evaluation")
parser.add_argument("--cp2k", help="CP2K evaluation")
parser.add_argument("--plot", help="plot evaluation")
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

def eval(cycle, **kwargs):
    eval_dir = Path.cwd() / '../qbc_train' / 'cycle{}'.format(cycle) / 'eval'
    assert eval_dir.exists()
    os.system('module swap cluster/accelgor; qstat')
    p = eval_dir.glob('*')
    files = [x for x in p]
    df = pd.DataFrame(index=sorted([x.name for x in files]), columns=[str(i) for i in range(5)])
    for f in files:
        p = f.glob('error*')
        errors = [x for x in p]
        for error in errors:
            with open(error, 'r') as e:
                line = e.readlines()[-1][:15].strip()
            df[error.name[-5]][f.name] = line
            if ('FileExistsError' in line) and kwargs['restart']:
                print('{} eval not finished, restarting ...'.format(error.name))
                os.system('module swap cluster/accelgor; cd {} ;qsub job{}.sh -d $(pwd)'.format(f, error.name[-5]))
            elif 'cuda' in line:
                with open(f / 'trajs' / '{}.log'.format(error.name[-5]) , 'r') as g:
                    ln = g.readlines()[-1][:6] + ' ps'
                df[error.name[-5]][f.name] = ln

            if (f / 'eval{}.xyz'.format(error.name[-5])).is_file():
                outp = len(ase.io.read(f / 'eval{}.xyz'.format(error.name[-5]), index=':'))
                df[error.name[-5]][f.name] = 'CP2K {}/20'.format(outp)
            
            elif (f / 'cp2k{}'.format(error.name[-5]) / 'error.txt').is_file():
                df[error.name[-5]][f.name] = 'CP2K started'

    print('EVAL:')
    print(df)

    if kwargs['cp2k']:
        config = Config.from_file(str(Path.cwd() / '../qbc_train' / 'cycle{}'.format(cycle) / 'params.yaml'), 'yaml')
        for f in files:
            p = f.glob('trajectory*')
            trajs = [x for x in p]
            for traj in trajs:
                cp2k_dir = f / 'cp2k{}'.format(traj.name[-5])
                if not cp2k_dir.exists():
                    cp2k_dir.mkdir()
                with open(cp2k_dir / 'job.sh','w') as rsh:
                    rsh.write(
                        '#!/bin/sh'
                        '\n\n#PBS -o output.txt'
                        '\n#PBS -e error.txt'
                        '\n#PBS -l walltime={}'
                        '\n#PBS -l nodes=1:ppn={}'
                        '\n\nsource ~/.cp2kenv'
                        '\npython ../../../../../QbC/util/cp2k_eval.py {} {} {}'.format(config.cp2k_walltime, config.cp2k_cores, str(traj), str(f / 'eval{}.xyz'.format(traj.name[-5])), config.cp2k_cores)
                    )
                os.system('module swap cluster/{}; cd {} ;qsub job.sh -d $(pwd)'.format(config.cp2k_cluster, cp2k_dir))

    if kwargs['plot']:
        E = pd.DataFrame(index=sorted([x.name for x in files]), columns=[str(i) for i in range(5)])
        F = pd.DataFrame(index=sorted([x.name for x in files]), columns=[str(i) for i in range(5)])
        S = pd.DataFrame(index=sorted([x.name for x in files]), columns=[str(i) for i in range(5)])
        F_mae = []
        temps = []
        vols = []
        for f in sorted(files):
            p = f.glob('eval*')
            trajs = [x for x in p]
            logging.info(f.name)
            for traj in sorted(trajs):
                gt = ase.io.read(traj, index=':', format='extxyz')
                if len(gt) == 20:
                    logging.info(traj.name[-5])
                    idcs = np.append(np.arange(1500, 20000, 2000), np.arange(1900, 20000, 2000))
                    pred = [ase.io.read(f / 'trajectory{}.xyz'.format(traj.name[-5]), index=i, format='extxyz') for i in idcs]
                    E_gt = np.array([x.get_potential_energy() for x in gt])
                    E_pred = np.array([x.get_potential_energy() for x in pred])
                    logging.info(E_gt.shape)
                    logging.info(E_pred.shape)
                    E[traj.name[-5]][f.name] = np.abs(E_gt - E_pred).mean()

                    F_gt = np.array([x.get_forces() for x in gt])
                    F_pred = np.array([x.get_forces() for x in pred])
                    logging.info(F_gt.shape)
                    logging.info(F_pred.shape)
                    F[traj.name[-5]][f.name] = np.abs(F_gt - F_pred).mean()

                    S_gt = np.array([x.get_stress() for x in gt])
                    S_pred = np.array([x.get_stress() for x in pred])
                    logging.info(S_gt.shape)
                    logging.info(S_pred.shape)
                    S[traj.name[-5]][f.name] = np.abs(S_gt - S_pred).mean()

                    F_mae += list(np.abs(F_gt - F_pred).mean((-1, -2)))
                    logging.info(F_mae)
                    temps += [x.get_temperature() for x in gt]
                    vols += [x.get_volume() for x in gt]
        logging.info('ENERGY:')
        logging.info(E)
        E.to_csv('E_mae.csv')

        logging.info('\nFORCES:')
        logging.info(F)
        F.to_csv('F_mae.csv')

        logging.info('\nSTRESS:')
        logging.info(S)
        S.to_csv('S_mae.csv')

        np.save('f_mae.npy', np.array(F_mae))
        np.save('temps.npy', np.array(temps))
        np.save('vols.npy', np.array(vols))

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
        
elif args.function == 'eval':
    eval(int(args.cycle), restart = args.restart, cp2k=args.cp2k, plot=args.plot)
elif args.function == 'log':
    log(int(args.cycle))
else:
    print('No valid command')