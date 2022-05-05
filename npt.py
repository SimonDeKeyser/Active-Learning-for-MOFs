from time import perf_counter
start_time = perf_counter()
import numpy as np
import scipy
import argparse
from pathlib import Path
import os

import pandas as pd
import ase.io
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import kB
from scipy.ndimage.filters import uniform_filter1d

import torch

from nequip.ase import NequIPCalculator
from nequip.utils import Config

import logging
logging.basicConfig(format='',level=logging.INFO)

from yaff_mtd import simulate
from qbc import QbC, CNNP
try:
    import ssh_keys
except ImportError:
    pass 
from vsc_shell import VSC_shell

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from cycler import cycler
plt.style.use('default')
plt.rcParams['axes.prop_cycle'] = cycler(marker= ['o', 'D', 'X']) * cycler(color= ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']) 
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams["legend.frameon"] = False
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.linestyle'] = '--'

plt.rcParams['axes.linewidth'] = 1
plt.rcParams['grid.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 2.

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

parser = argparse.ArgumentParser(
            description="Create MD trajectory and select for Query by Committee cycle."
        )
parser.add_argument("model", help='Path to deployed model .pth file')
parser.add_argument("atoms", help='Path to optimized atoms .xyz file')
parser.add_argument("n_select", help="The amount of structures to select with QbC")
args = parser.parse_args()

##########################################################################################

MD_runs = 5
TP = [(300,0), (350,0), (400,0), (300,1), (350,1)]

##########################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
calc = NequIPCalculator.from_deployed_model(
        model_path=args.model,
        device=device
    )
init = ase.io.read(args.atoms, index=0, format='extxyz')
n_select = int(args.n_select)

MD_steps = int(np.load(Path.cwd() / '../' / 'MD_steps.npy')[0])
logging.info('MD steps: {}'.format(MD_steps))

traj_dir = Path.cwd() / 'trajs'
if not traj_dir.exists():
    traj_dir.mkdir()
    completed = 0
else:
    logging.info('This is a restarted run, probably because walltime exceeded')
    p = traj_dir.glob('*.xyz')
    traj_files = [x for x in p] 
    completed = len(traj_files)

if Path('fails.npy').exists():
    fails = np.load('fails.npy')[0]
else:
    fails = 0       

for i in range(MD_runs)[completed:]:
    atoms = init.copy()
    atoms.set_calculator(calc=calc)
    T, P = TP[i]
    MaxwellBoltzmannDistribution(atoms=atoms, temp=T * kB)
    try:
        simulate(MD_steps, 10, 0, atoms, calc, T, P, path_h5= traj_dir / '{}.h5'.format(i), path_xyz= traj_dir / '{}.xyz'.format(i))
    except:
        logging.info('MD failed')
        fails += 1

np.save('fails.npy', np.array([fails]))
logging.info('Fails: {}'.format(fails))    

select_per_MD = n_select // MD_runs
p = traj_dir.glob('*.xyz')
traj_files = [x for x in p]
new_datapoints = []
scores = []
for traj_file in sorted(traj_files):
    print(traj_file.name)
    try:
        traj = ase.io.read(traj_file, index=':', format='extxyz')
    except:
        pass
    n_atom =  len(traj[0])
    energies = np.array([atoms.info['energy']/n_atom for atoms in traj])
    volumes = np.array([atoms.cell.volume/4 for atoms in traj])

    filter_n = len(traj)//10
    MA = uniform_filter1d(energies, filter_n, mode= 'nearest')
    
    std = 2.5*(np.array(energies)[filter_n:] - MA[filter_n:]).std()
    
    score = (np.abs(energies[filter_n:]-MA[filter_n:])/std).mean()
    scores.append(score)
    condition = np.arange(len(energies) - filter_n)[np.abs(energies[filter_n:]-MA[filter_n:]) < std]
    data = [traj[filter_n:][i] for i in condition]
    logging.info('Valid data length: {}'.format(len(data)))
    start = len(data)//(select_per_MD+1)
    stop = len(data) - start
    selected_idc = np.linspace(start,stop, select_per_MD).astype(int)
    new_datapoints += [data[i] for i in selected_idc]

    selected_idc_e = np.arange(len(energies)-filter_n)[condition[np.arange(len(condition))[selected_idc]]]

    time = np.arange(len(energies))*0.5*10*1e-3
    fig, axs = plt.subplots(3, figsize=(10,8))
    axs[0].plot(time, energies, '.--' ,color='k')
    axs[0].plot(time[filter_n:], MA[filter_n:],'-', label='MA({})'.format(filter_n), color='red', alpha=0.8)
    axs[0].fill_between(time[filter_n:], MA[filter_n:] - std, MA[filter_n:] + std, label='$\sigma$(E/N - MA)', color='blue', alpha=0.2)
    axs[0].fill_between(time[filter_n:], MA[filter_n:] - 2.5*std, MA[filter_n:] + 2.5*std, label='2.5$\sigma$(E/N - MA)', color='green', alpha=0.2)
    axs[0].plot(time[filter_n:][selected_idc_e], energies[filter_n:][selected_idc_e], 'o' ,color='blue', label='Selected')
    axs[0].set_ylabel('$E/N$ [eV/atom]')
    axs[0].set_xticklabels([])
    axs[1].plot(time[filter_n:], -100*(np.array(energies[filter_n:])/np.array(MA[filter_n:]) - 1), '.--' ,color='k')
    axs[1].plot(time[filter_n:], np.zeros(len(time)-filter_n) ,'-', color='red', alpha=0.8)
    axs[1].fill_between(time[filter_n:], -100*std/MA[filter_n:], +100*std/MA[filter_n:], color='blue', alpha=0.2)
    axs[1].fill_between(time[filter_n:], -100*2.5*std/MA[filter_n:], +100*2.5*std/MA[filter_n:], color='green', alpha=0.2)
    axs[1].plot(time[filter_n:][selected_idc_e], -100*(np.array(energies[filter_n:][selected_idc_e])/np.array(MA[filter_n:][selected_idc_e]) - 1), 'o' ,color='blue')
    axs[1].set_ylabel('Relative Deviation [%]'.format(filter_n))
    axs[1].set_xlim(axs[0].get_xlim())
    axs[1].set_xticklabels([])
    axs[2].set_xlabel('Time [ps]')
    axs[2].plot(time, volumes/volumes[0], '.--' ,color='k')
    axs[2].plot(time[filter_n:][selected_idc_e], volumes[filter_n:][selected_idc_e]/volumes[0], 'o' ,color='blue')
    axs[2].set_ylabel('$V/V_0$')
    fig.legend(loc=7)
    fig.tight_layout()
    fig.subplots_adjust(right=0.75)
    plt.savefig(traj_file.parent / '{}.png'.format(traj_file.name[:-4]), bbox_inches = 'tight')

np.save(Path.cwd() / 'score.npy', np.array(scores))
ase.io.write(Path.cwd() / 'new_data.xyz', new_datapoints, format='extxyz')

logging.info('WALLTIME: [h]')
logging.info((perf_counter() - start_time)/3600)

cp2k_dir = Path.cwd() / 'cp2k'
if not cp2k_dir.exists():
    cp2k_dir.mkdir()

config = Config.from_file(str(Path.cwd() / '../../' / 'params.yaml'), 'yaml')
with open(cp2k_dir / 'job.sh','w') as rsh:
    rsh.write(
        '#!/bin/sh'
        '\n\n#PBS -o output.txt'
        '\n#PBS -e error.txt'
        '\n#PBS -l walltime={}'
        '\n#PBS -l nodes=1:ppn={}'
        '\n\nsource ~/.cp2kenv'
        '\npython ../../../../../QbC/cp2k_main.py {} {} {}'.format(config.cp2k_walltime, config.cp2k_cores, str(Path.cwd() / 'new_data.xyz'), str(Path.cwd() / 'calculated_data.xyz'), config.cp2k_cores)
    )

shell = VSC_shell(ssh_keys.HOST, ssh_keys.USERNAME, ssh_keys.PASSWORD, ssh_keys.KEY_FILENAME)
shell.submit_job(config.cp2k_cluster, cp2k_dir.resolve(), 'job.sh')
shell.__del__()