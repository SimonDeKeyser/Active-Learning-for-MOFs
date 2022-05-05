from calendar import c
import sys
import torch
import time
import numpy as np
from pathlib import Path
import h5py
import yaff
from yaff.external.libplumed import ForcePartPlumed
import molmod

from ase.io import read, write
from ase import Atoms
from ase.geometry import Cell
from ase.stress import voigt_6_to_full_3x3_stress
from nequip.ase import NequIPCalculator
import logging
logging.basicConfig(format='',level=logging.INFO)

import functools
print = functools.partial(print, flush=True)

# when evaluating a model that was converted to double precision, the default
# dtype of torch must be changed as well; otherwise the data loading within
# the calculator will still be performed in single precision
#torch.set_default_dtype(torch.float64)

class ExtXYZHook(yaff.sampling.iterative.Hook):

    def __init__(self, path_xyz, step=10, start=0):
        super().__init__(step=step, start=start)
        if Path(path_xyz).exists():
            Path(path_xyz).unlink() # remove if exists
        self.path_xyz = path_xyz
        self.atoms = None

    def init(self, iterative):
        self.atoms = Atoms(
                numbers=iterative.ff.system.numbers.copy(),
                positions=iterative.ff.system.pos / molmod.units.angstrom,
                cell=iterative.ff.system.cell._get_rvecs() / molmod.units.angstrom,
                pbc=True,
                )

    def pre(self, iterative):
        pass

    def post(self, iterative):
        pass

    def __call__(self, iterative):
        if self.atoms is None:
            self.init(iterative)
        self.atoms.set_positions(iterative.ff.system.pos / molmod.units.angstrom)
        cell = iterative.ff.system.cell._get_rvecs() / molmod.units.angstrom
        self.atoms.set_cell(cell)
        self.atoms.arrays['forces'] = -iterative.ff.gpos * molmod.units.angstrom / molmod.units.electronvolt
        self.atoms.info['energy'] = iterative.ff.energy / molmod.units.electronvolt
        volume = np.linalg.det(cell)
        self.atoms.info['stress'] = iterative.ff.vtens / (molmod.units.electronvolt * volume)
        write(self.path_xyz, self.atoms, append=True)

class ForcePartASE(yaff.pes.ForcePart):
    """YAFF Wrapper around an ASE calculator"""

    def __init__(self, system, atoms, calculator):
        """Constructor

        Parameters
        ----------

        system : yaff.System
            system object

        atoms : ase.Atoms
            atoms object with calculator included.

        """
        yaff.pes.ForcePart.__init__(self, 'ase', system)
        self.system = system # store system to obtain current pos and box
        self.atoms  = atoms
        self.calculator = calculator

    def _internal_compute(self, gpos=None, vtens=None):
        self.atoms.set_positions(self.system.pos / molmod.units.angstrom)
        self.atoms.set_cell(Cell(self.system.cell._get_rvecs() / molmod.units.angstrom))
        energy = self.atoms.get_potential_energy() * molmod.units.electronvolt
        if gpos is not None:
            forces = self.atoms.get_forces()
            gpos[:] = -forces * molmod.units.electronvolt / molmod.units.angstrom
        if vtens is not None:
            volume = np.linalg.det(self.atoms.get_cell())
            stress = voigt_6_to_full_3x3_stress(self.atoms.get_stress())
            vtens[:] = volume * stress * molmod.units.electronvolt
        return energy

def create_forcefield(atoms, calculator):
    """Creates force field from ASE atoms instance"""
    system = yaff.System(
            numbers=atoms.get_atomic_numbers(),
            pos=atoms.positions * molmod.units.angstrom,
            rvecs=atoms.get_cell() * molmod.units.angstrom,
            )
    system.set_standard_masses()
    part_ase = ForcePartASE(system, atoms, calculator)
    return yaff.pes.ForceField(system, [part_ase])


def simulate(steps, step, start, atoms, calculator, temperature, pressure=None, path_h5=None, path_xyz=None):
    """Samples the phase space using Langevin dynamics"""
    # set default output paths
    if path_h5 is None:
        path_h5  = Path.cwd() / 'md.h5'
    if path_xyz is None:
        path_xyz = Path.cwd() / 'md.xyz'

    # create forcefield from atoms
    ff = create_forcefield(atoms, calculator)

    # hooks
    hooks = []
    plu = ForcePartPlumed(ff.system, timestep= 0.5*molmod.units.femtosecond, fn='plumed.dat')
    ff.add_part(plu)
    hooks.append(plu)

    loghook = yaff.VerletScreenLog(step=step, start=0)
    hooks.append(loghook)
    if path_h5 is not None:
        h5file = h5py.File(path_h5, 'w')
        h5hook = yaff.HDF5Writer(h5file, step=step, start=start)
        hooks.append(h5hook)
    if path_xyz is not None:
        xyzhook = ExtXYZHook(str(path_xyz), step=step, start=start)
        hooks.append(xyzhook)

    # temperature / pressure control
    thermo = yaff.LangevinThermostat(temperature, timecon=100 * molmod.units.femtosecond)
    if pressure is None:
        print('CONSTANT TEMPERATURE, CONSTANT VOLUME')
        #vol_constraint = True
        #pressure = 0 # dummy pressure
        hooks.append(thermo)
    else:
        print('CONSTANT TEMPERATURE, CONSTANT PRESSURE')
        vol_constraint = False
        baro = yaff.LangevinBarostat(
                ff,
                temperature,
                pressure * 1e6 * molmod.units.pascal, # pressure in MPa
                timecon=molmod.units.picosecond,
                anisotropic=True,
                vol_constraint=vol_constraint,
                )
        tbc = yaff.TBCombination(thermo, baro)
        hooks.append(tbc)

    # integration
    verlet = yaff.VerletIntegrator(
            ff,
            timestep=0.5*molmod.units.femtosecond,
            hooks=hooks,
            temp0=temperature, # initialize velocities to correct temperature
            )
    yaff.log.set_level(yaff.log.medium)
    logging.info('Starting run...')
    verlet.run(steps)
    yaff.log.set_level(yaff.log.silent)

if __name__ == '__main__':
    steps = 2000
    step  = 10
    start = 0
    temperature = float(sys.argv[2])
    pressure    = 1
    path_model  = str(sys.argv[3])
    path_atoms  = str(sys.argv[4])

    with open(path_atoms, 'r') as f:
        atoms = read(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calculator = NequIPCalculator.from_deployed_model(path_model, device=device)
    atoms.set_calculator(calc=calculator)

    simulate(steps, step, start, atoms, calculator, temperature, pressure)

