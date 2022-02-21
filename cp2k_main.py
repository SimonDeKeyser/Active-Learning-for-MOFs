from pathlib import Path

import ase
from ase.io import read, write
from ase.io.extxyz import write_extxyz
from cp2k_calculator import CP2K
#from ase.io.trajectory import Trajectory
from ase.stress import voigt_6_to_full_3x3_stress

from ase.calculators.loggingcalc import LoggingCalculator
#from ase.optimize.precon import Exp, PreconLBFGS
#from ase.constraints import ExpCellFilter


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

atoms = read(Path.cwd() / 'chunk.xyz')
calculator = CP2K(
        atoms=atoms,
        auto_write=True,
        basis_set=None,
        command='mpirun -np 24 cp2k_shell.psmp',
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