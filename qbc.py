from dataclasses import dataclass
import logging
import random
from pathlib import Path

logging.basicConfig(format='',level=logging.INFO)

import matplotlib.pyplot as plt
plt.style.use('default')
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color= ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e'])
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
plt.rcParams['mathtext.fontset'] = 'cm'
import numpy as np
import ase.io
import torch
from nequip.scripts.deploy import load_deployed_model
from nequip.data import AtomicData

@dataclass
class qbc:
    """
    Class containing the Query by Committee
    """
    name: str
    models_dir: str
    traj_dir: str
    results_dir: str
    traj_index: str = ':'
    n_select: int = 100
    nequip_train: bool = False
    r_max: float = 4.5

    def __post_init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        logging.info('Using {} device'.format(self.device))
        assert self.models_dir.is_dir(), 'The models directory does not exist'
        assert self.traj_dir.exists(), 'The trajectory file does not exist'
        self.results_dir = self.results_dir / self.name
        if not self.results_dir.exists():
            self.results_dir.mkdir()
            logging.info('Created results directory: \n{}'.format(self.results_dir))
        self.store_models()
        self.load_traj()
        self.traj_e = np.zeros(self.traj_len)
        self.mean_e, self.sig_e = np.zeros(self.traj_len), np.zeros(self.traj_len)
        self.mean_f_mean, self.sig_f_mean = np.zeros(self.traj_len), np.zeros(self.traj_len)
        self.mean_f_max, self.sig_f_max = np.zeros(self.traj_len), np.zeros(self.traj_len)
        assert self.n_select<self.traj_len, 'The wanted amount of selection datapoints exceeds the amount of total datapoints: {}>{}'.format(self.n_select, self.traj_len)

    def store_models(self):
        self.models = {}
        p = Path(self.models_dir).glob('**/*')
        if self.nequip_train:
            model_files = [x for x in p if x.is_dir()]
            assert 'processed' in [f.name for f in model_files], 'The given models directory is not a NequIP training directory'
            logging.info('Models found in the NequIP training directory:')
            for file in sorted(model_files):
                if not file.name == 'processed':
                    logging.info('*\t{}'.format(file.name))
                    path = file / 'best_model.pth'
                    model = torch.load(path, map_location=self.device)
                    model = model.to(self.device)
                    self.models[file.name] = model  
        else:
            model_files = [x for x in p if x.is_file()]
            logging.info('Models found in the models directory:')
            for file in model_files:
                if file.suffix == '.pth':
                    logging.info('*\t{}'.format(file.name[:-4]))
                    model, _ = load_deployed_model(file, self.device)
                    assert(model.__class__.__name__ == 'RecursiveScriptModule')
                    self.models[file.name[:-4]] = model       
        self.models_len = len(self.models)
        assert self.models_len >= 2, 'The amount fo models in the committee is less than 2'
        logging.info('\n')
    
    def load_traj(self):
        logging.info('Loading trajectory from ...: \n{}'.format(self.traj_dir))
        assert(self.traj_dir.suffix == '.xyz')
        self.traj = ase.io.read(self.traj_dir, index=self.traj_index, format='extxyz')
        self.traj_len = len(self.traj)
        logging.info('... Trajectory loaded\n')

    def predict(self, atoms):
        data = AtomicData.from_ase(atoms=atoms, r_max=self.r_max)
        data = data.to(self.device)
        energies = np.zeros(self.models_len)
        forces = np.zeros((self.models_len,len(atoms),3))
        j = 0
        for model in self.models.values():
            out = model(AtomicData.to_AtomicDataDict(data))
            energies[j] = out['total_energy'].detach().cpu().item()
            forces[j,:,:] = out['forces'].detach().cpu().numpy()
            j += 1
            
        return energies, forces

    def disagreement(self, pred, prop, red=None):
        mean = np.mean(pred, axis=0)
        std = np.std(pred, axis=0)
        if prop == 'forces':
            if red == 'mean':
                mean = np.mean(abs(mean),axis=(0,1))
                std = np.mean(std,axis=(0,1))
            if red == 'max':
                mean = np.max(abs(mean),axis=(0,1))
                std = np.max(std,axis=(0,1))
        
        return mean, std

    def evaluate_committee(self, save=False):
        logging.info('Starting evaluation ...')
        for i in range(self.traj_len):
            self.traj_e[i] = self.traj[i].get_potential_energy()
            energies, forces = self.predict(self.traj[i])
            self.mean_e[i], self.sig_e[i] = self.disagreement(energies, 'energy')
            self.mean_f_mean[i], self.sig_f_mean[i] = self.disagreement(forces, 'forces', 'mean')
            self.mean_f_max[i], self.sig_f_max[i] = self.disagreement(forces, 'forces', 'max')
            if (i%10) == 0:
                logging.info('[{}/{}]'.format(i,self.traj_len))
        logging.info('... Evaluation finished\n')
        if save:
            self.save()

    def select_data(self, prop, red=None):
        if prop != 'random':
            if prop == 'energy':
                disagreement = self.sig_e
            if prop == 'forces':
                if red == 'mean':
                    disagreement = self.sig_f_mean
                if red == 'max':
                    disagreement = self.sig_f_max 
            
            assert disagreement is not None, 'No valid disagreement metric was given'
            part = np.argpartition(disagreement, -self.n_select)
            ind = np.array(part[-self.n_select:])
        else:
            ind = random.sample(range(len(self.traj)),self.n_select)
        assert ind is not None, 'No valid disagreement metric was given'
        selected_data = [self.traj[i] for i in ind]
        self.ind = ind
        return selected_data

    def save(self):
        logging.info('Saving results in ...: \n{}'.format(self.results_dir))
        np.save(self.results_dir / 'traj_e.npy', self.traj_e)
        np.save(self.results_dir / 'mean_e.npy', self.mean_e)
        np.save(self.results_dir / 'mean_f_mean.npy', self.mean_f_mean)    
        np.save(self.results_dir / 'mean_f_max.npy', self.mean_f_max)
        np.save(self.results_dir / 'sig_e.npy', self.sig_e)
        np.save(self.results_dir / 'sig_f_mean.npy', self.sig_f_mean)    
        np.save(self.results_dir / 'sig_f_max.npy', self.sig_f_max)
        logging.info('... Results saved\n')
    
    def load(self):
        logging.info('Loading results from ...: \n{}'.format(self.results_dir))
        self.traj_e = np.load(self.results_dir / 'traj_e.npy')
        self.mean_e = np.load(self.results_dir / 'mean_e.npy')
        self.mean_f_mean = np.load(self.results_dir / 'mean_f_mean.npy')    
        self.mean_f_max = np.load(self.results_dir / 'mean_f_max.npy')
        self.sig_e = np.load(self.results_dir / 'sig_e.npy')
        self.sig_f_mean = np.load(self.results_dir / 'sig_f_mean.npy')    
        self.sig_f_max = np.load(self.results_dir / 'sig_f_max.npy')
        logging.info('... Results loaded\n')        

    def plot_traj_disagreement(self, from_results_dir=False):
        assert self.ind is not None, 'First select indices to plot them'
        if from_results_dir:
            self.load()
        ylabel1 = '$E_{pred}$ [eV]'
        ylabel2 = '$\sigma_{E}$ [eV]'
        ylabel3 = '$\overline{\sigma_{\mathbf{F}}}$ [eV/$\AA$]'
        ylabel4 = '$max(\sigma_{\mathbf{F}})$ [eV/$\AA$]'
        time = np.arange(self.traj_len)

        fig, axs = plt.subplots(4, figsize=(10,12), gridspec_kw = {'wspace':0, 'hspace':0.05})
        axs[0].plot(time, self.traj_e, '-.r',markersize=4, label='Trajectory')
        axs[0].plot(time, self.mean_e, '.k',markersize=4, label='$\overline{NNPs}$')
        axs[0].set_ylabel(ylabel1)
        axs[0].set_xticklabels([])
        axs[0].legend()
        axs[0].grid()
        
        axs[1].plot(time, self.sig_e,'.k', markersize=4)
        axs[1].plot(time[self.ind],self.sig_e[self.ind],'.', color='red', markersize=4, label='{} selected $\sigma$'.format(self.n_select))
        axs[1].set_ylabel(ylabel2)
        axs[1].grid()
        axs[1].set_xticklabels([])
        axs[1].legend()
        
        axs[2].plot(time, self.sig_f_mean,'.k', markersize=4)
        axs[2].plot(time[self.ind],self.sig_f_mean[self.ind],'.', color='red', markersize=4, label='{} selected $\sigma$'.format(self.n_select))
        axs[2].set_ylabel(ylabel3)
        axs[2].grid()
        axs[2].set_xticklabels([])
        axs[2].legend()
        
        axs[3].plot(time, self.sig_f_max,'.k', markersize=4)
        axs[3].plot(time[self.ind],self.sig_f_max[self.ind],'.', color='red', markersize=4, label='{} selected $\sigma$'.format(self.n_select))
        axs[3].set_ylabel(ylabel4)
        axs[3].grid()
        axs[3].set_xlabel('Step')
        axs[3].legend()

        plt.savefig('{}/traj_disagreement'.format(self.results_dir),dpi=100)