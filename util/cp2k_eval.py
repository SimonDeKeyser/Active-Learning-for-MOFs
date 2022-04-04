from pathlib import Path
import argparse
import os
import sys
p = os.path.abspath('/scratch/gent/vo/000/gvo00003/vsc43785/Thesis/q4_MOFs/QbC')
sys.path.insert(1,p)

import ase
from ase.io import read
from ase.io.extxyz import write_extxyz
from cp2k_calculator import CP2K
from ase.stress import voigt_6_to_full_3x3_stress

import numpy as np

import logging
logging.basicConfig(format='',level=logging.INFO)

path_source = Path('/scratch/gent/437/vsc43785/cp2k') / 'SOURCEFILES'
path_potentials = path_source / 'GTH_POTENTIALS'
path_basis      = path_source / 'BASISSETS'
path_dispersion = path_source / 'dftd3.dat'

parser = argparse.ArgumentParser(
        description="Calculate new datapoints."
    )
parser.add_argument("input_dir", help="Path to .xyz file to be calculated")
parser.add_argument("output_dir", help="Path where the calculated .xyz file should be written to")
parser.add_argument("cores", help="Amount of cores in HPC job")
parser.add_argument("--mult", help="Amount of cores in HPC job")
args = parser.parse_args()

if args.mult is None:
    mult = 1
    uks = False
else:
    mult = args.mult
    uks = True

additional_input = '''
 &FORCE_EVAL
   &DFT
     BASIS_SET_FILE_NAME  {}
     POTENTIAL_FILE_NAME  {}
     MULTIPLICITY  {}
     &SCF
       MAX_SCF  25
       MAX_DIIS  8
       EPS_SCF  1.0E-06
       SCF_GUESS  RESTART
       &OT
         MINIMIZER  CG
         PRECONDITIONER  FULL_SINGLE_INVERSE
       &END OT
       &OUTER_SCF  T
         MAX_SCF  25
         EPS_SCF  1.0E-06
       &END OUTER_SCF
       &PRINT
         &RESTART
           ADD_LAST  SYMBOLIC
           &EACH
             GEO_OPT  100
             CELL_OPT 100
             MD  1
           &END EACH
         &END RESTART
       &END PRINT
     &END SCF
     &QS
       METHOD  GPW
       EPS_DEFAULT  1.0E-10
       EXTRAPOLATION  USE_GUESS
     &END QS
     &MGRID
       ! CUTOFF [Ry]  800.0
       REL_CUTOFF [Ry]  60.0
       NGRIDS  5
     &END MGRID
     &XC
       DENSITY_CUTOFF   1.0E-10
       GRADIENT_CUTOFF  1.0E-10
       TAU_CUTOFF       1.0E-10
       &XC_FUNCTIONAL  PBE
       &END XC_FUNCTIONAL
       &VDW_POTENTIAL
         POTENTIAL_TYPE  PAIR_POTENTIAL
         &PAIR_POTENTIAL
           TYPE  DFTD3(BJ)
           PARAMETER_FILE_NAME  {}
           REFERENCE_FUNCTIONAL PBE
           R_CUTOFF  25
         &END PAIR_POTENTIAL
       &END VDW_POTENTIAL
     &END XC
   &END DFT
   &SUBSYS
     &KIND Al
       ELEMENT  Al
       BASIS_SET TZVP-MOLOPT-SR-GTH
       POTENTIAL GTH-PBE-q3
     &END KIND
     &KIND Ga
       ELEMENT  Ga
       BASIS_SET TZVP-MOLOPT-SR-GTH
       POTENTIAL GTH-PBE-q13
     &END KIND
     &KIND V
       ELEMENT  V
       BASIS_SET TZVP-MOLOPT-SR-GTH
       POTENTIAL GTH-PBE-q13
     &END KIND
     &KIND O
       ELEMENT  O
       BASIS_SET TZVP-MOLOPT-GTH
       POTENTIAL GTH-PBE-q6
     &END KIND
     &KIND C
       ELEMENT  C
       BASIS_SET TZVP-MOLOPT-GTH
       POTENTIAL GTH-PBE-q4
     &END KIND
     &KIND F
       ELEMENT  F
       BASIS_SET TZVP-MOLOPT-GTH
       POTENTIAL GTH-PBE-q7
     &END KIND
     &KIND H
       ELEMENT  H
       BASIS_SET TZVP-MOLOPT-GTH
       POTENTIAL GTH-PBE-q1
     &END KIND
   &END SUBSYS
 &END FORCE_EVAL
        '''.format(path_basis, path_potentials, mult, path_dispersion)

input_dir = args.input_dir
output_dir = args.output_dir
cores = int(args.cores)

atoms = read(input_dir)
calculator = CP2K(
        atoms=atoms,
        auto_write=True,
        basis_set=None,
        command='mpirun -np {} cp2k_shell.psmp'.format(cores),
        cutoff=1000 * ase.units.Rydberg,
        stress_tensor=True,
        uks=uks,
        print_level='LOW',
        inp=additional_input,
        pseudo_potential=None,
        max_scf=None,           # disable
        xc=None,                # disable
        basis_set_file=None,    # disable
        charge=None,            # disable
        potential_file=None,    # disable
        )
atoms.calc = calculator

#idcs = np.arange(1500, 20000, 2000)
#idcs = np.arange(1900, 20000, 2000)
idcs = np.arange(1000, 4700, 400)
logging.info('input: {}'.format(input_dir))
chunk = [read(input_dir, index=i) for i in idcs]

if Path(output_dir).is_file():  
    logging.info('This is a restarted run, probably because walltime exceeded...')
    calc_data = ase.io.read(output_dir, index=':')
    complete = len(calc_data)
    logging.info('Restarting calculation, {} already complete'.format(complete))
else:
    calc_data = []
    complete = 0

for state in chunk[complete:]:
    state.calc = None
    atoms.set_positions(state.get_positions())
    atoms.set_cell(state.get_cell())
    state.arrays['forces'] = atoms.get_forces()
    state.info['stress'] = voigt_6_to_full_3x3_stress(atoms.get_stress())
    state.info['energy'] = atoms.get_potential_energy()
    calc_data.append(state)
    with open(output_dir, 'w') as f:
        write_extxyz(f, calc_data)
    logging.info('Energy: {} eV'.format(state.info['energy']))
