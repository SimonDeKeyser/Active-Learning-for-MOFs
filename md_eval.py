import numpy as np
import scipy
import argparse
from pathlib import Path
import os

import pandas as pd
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

parser = argparse.ArgumentParser(
            description="Create MD trajectory and select for Query by Committee cycle."
        )
parser.add_argument("model", help='Path to deployed model .pth file')
parser.add_argument("atoms", help='Path to optimized atoms .xyz file')
parser.add_argument("i", help='Index of trajectory')
args = parser.parse_args()

##########################################################################################

MD_steps = 200000
S_eps = 0
T = 300
dt = 0.5

##########################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(device)
calc = NequIPCalculator.from_deployed_model(
        model_path=args.model,
        device=device
    )
init = ase.io.read(args.atoms, index=0, format='extxyz')

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

S = make_sym_matrix(np.random.uniform(-S_eps, S_eps, 6))
atoms = strain_cell_sampling(init, S)
atoms.set_calculator(calc=calc)

MaxwellBoltzmannDistribution(atoms=atoms, temp=T * kB)
dyn = Langevin(atoms, timestep=dt * fs, friction = 0.002,
        temperature_K = T,
        trajectory= str(traj_dir / '{}.traj'.format(args.i)), 
        logfile= str(traj_dir / '{}.log'.format(args.i)), loginterval=10
        )
dyn.run(MD_steps)

traj = Trajectory(Path.cwd() / 'trajs' / '{}.traj'.format(args.i))
ase.io.write(Path.cwd() / 'trajectory{}.xyz'.format(args.i), traj, format='extxyz')


