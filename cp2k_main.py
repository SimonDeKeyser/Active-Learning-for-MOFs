from pathlib import Path
import argparse
import os

from nequip.utils import Config
import ase
from ase.io import read, write
from ase.io.extxyz import write_extxyz
from cp2k_calculator import CP2K
#from ase.io.trajectory import Trajectory
from ase.stress import voigt_6_to_full_3x3_stress

from ase.calculators.loggingcalc import LoggingCalculator
import ssh_keys
from vsc_shell import VSC_shell
#from ase.optimize.precon import Exp, PreconLBFGS
#from ase.constraints import ExpCellFilter

import logging
logging.basicConfig(format='',level=logging.INFO)

path_source = Path('/scratch/gent/437/vsc43785/cp2k') / 'SOURCEFILES'
path_potentials = path_source / 'GTH_POTENTIALS'
path_basis      = path_source / 'BASISSETS'
path_dispersion = path_source / 'dftd3.dat'

additional_input = '''
 &FORCE_EVAL
   &DFT
     BASIS_SET_FILE_NAME  {}
     POTENTIAL_FILE_NAME  {}
     UKS  F
     MULTIPLICITY  1
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
     &KIND H
       ELEMENT  H
       BASIS_SET TZVP-MOLOPT-GTH
       POTENTIAL GTH-PBE-q1
     &END KIND
   &END SUBSYS
 &END FORCE_EVAL
        '''.format(path_basis, path_potentials, path_dispersion)

parser = argparse.ArgumentParser(
        description="Calculate new datapoints."
    )
parser.add_argument("input_dir", help="Path to .xyz file to be calculated")
parser.add_argument("output_dir", help="Path where the calculated .xyz file should be written to")
parser.add_argument("cores", help="Amount of cores in HPC job")
args = parser.parse_args()

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
        uks=False,
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

with open(input_dir,'r') as f:
    chunk = list(read(f, index=':'))

for state in chunk:
    state.calc = None
    atoms.set_positions(state.get_positions())
    atoms.set_cell(state.get_cell())
    state.arrays['forces'] = atoms.get_forces()
    state.info['stress'] = voigt_6_to_full_3x3_stress(atoms.get_stress())
    state.info['energy'] = atoms.get_potential_energy()

with open(output_dir, 'w') as f:
    write_extxyz(f, chunk)

md_dir = (Path.cwd() / '../../').resolve()
p = md_dir.glob('**/*/calculated_data.xyz')
n_ready = len([x for x in p])
p = md_dir.glob('*')
n_total = len([x for x in p if x.is_dir()])

if n_total == n_ready:
    logging.info('All QbCs are ready, restarting training...')
    config = Config.from_file(str(Path.cwd() / '../../../' / 'params.yaml'), 'yaml')
    runs_dir = (Path.cwd() / '../../../../../' / 'QbC' / 'runs').resolve()

    with open(runs_dir / 'cycle{}_restart.sh'.format(config.cycle),'w') as rsh:
        rsh.write(
            '#!/bin/sh'
            '\n\n#PBS -l walltime={}'
            '\n#PBS -l nodes=1:ppn={}:gpus=1'
            '\n\nsource ~/.{}'
            '\npython ../train.py {} {} --traj-index {} --cp2k-restart {}'.format(config.walltime, config.cores, config.env, config.cycle, config.walltime, config.traj_index, True)
        )

    shell = VSC_shell(ssh_keys.HOST, ssh_keys.USERNAME, ssh_keys.PASSWORD, ssh_keys.KEY_FILENAME)
    shell.submit_job(config.cluster, runs_dir, 'cycle{}_restart.sh'.format(config.cycle))
    shell.__del__()

else:
  logging.info('Only {} of {} QbC new training data is calculated, waiting...'.format(n_ready, n_total))