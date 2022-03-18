import numpy as np
import scipy
import argparse
from pathlib import Path

import ase.io
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import fs, kB, Pascal
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory

import torch

from nequip.ase import NequIPCalculator
from nequip.utils import Config

import logging
logging.basicConfig(format='',level=logging.INFO)

from qbc import QbC, CNNP
try:
    import ssh_keys
except ImportError:
    pass 
from vsc_shell import VSC_shell

parser = argparse.ArgumentParser(
            description="Create MD trajectory and select for Query by Committee cycle."
        )
parser.add_argument("model", help='Path to deployed model .pth file')
parser.add_argument("atoms", help='Path to optimized atoms .xyz file')
parser.add_argument("n_select", help="The amount of structures to select with QbC")
args = parser.parse_args()
##########################################################################################

MD_runs = 2
MD_steps = 100
eps = 5e-2
T = 300
dt = 1

##########################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
calc = NequIPCalculator.from_deployed_model(
        model_path=args.model,
        device=device
    )
init = ase.io.read(args.atoms, index=0, format='extxyz')
n_select = int(args.n_select)

def make_sym_matrix(vals):
    A = np.zeros((3,3))
    xs,ys = np.triu_indices(3)
    A[xs,ys] = vals
    A[ys,xs] = vals
    return A

def strain_cell_sampling(init, S):
    new = init.copy()
    scaled_pos = new.cell.scaled_positions(new.positions)
    sqrt = scipy.linalg.sqrtm(2*S+np.eye(3))
    new.cell.array = np.dot(new.cell.array, sqrt)
    new.positions = new.cell.cartesian_positions(scaled_pos)
    
    return new

traj_dir = Path.cwd() / 'trajs'
if not traj_dir.exists():
    traj_dir.mkdir()

for i in range(MD_runs):
    S = make_sym_matrix(np.random.uniform(-eps, eps, 6))
    atoms = strain_cell_sampling(init, S)
    atoms.set_calculator(calc=calc)
    
    MaxwellBoltzmannDistribution(atoms=atoms, temp=T * kB)
    #dyn = Langevin(atoms, timestep=dt * fs, friction = 0.002,
    #        temperature_K = T,
    #        trajectory= str(traj_dir / '{}.traj'.format(i)), 
    #        logfile= str(traj_dir / '{}.log'.format(i)), loginterval=10
    #        )

    #dyn.run(MD_steps)

p = traj_dir.glob('*.traj')
traj_files = [x for x in p]
total_traj = []
for traj_file in traj_files:
    logging.info('Adding trajectory {} for QbC'.format(traj_file.name[:-5]))
    traj = Trajectory(traj_file)
    total_traj += traj

ase.io.write(Path.cwd() / 'trajectory.xyz', total_traj, format='extxyz')

logging.info('Committee from {}'.format(Path(args.model).parents[1]))

cnnp = CNNP(Path(args.model).parents[1]).load_models_from_nequip_deployed()
if not (Path.cwd() / 'qbc').exists():
    (Path.cwd() / 'qbc').mkdir()

committee = QbC(results_dir=Path.cwd() / 'qbc', 
            cnnp=cnnp, 
            traj_dir=Path.cwd() / 'trajectory.xyz',
            n_select=n_select
        )

committee.evaluate_committee(save=True)
new_datapoints = committee.select_data(prop='forces', red='mean')
committee.plot_traj_disagreement()
ase.io.write(Path.cwd() / 'new_data.xyz', new_datapoints, format='extxyz')

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