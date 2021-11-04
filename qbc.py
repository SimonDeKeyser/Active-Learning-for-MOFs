import os
from dataclasses import dataclass
import logging
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
from nequip.data import AtomicData, AtomicDataDict

models_dir = '/scratch/gent/vo/000/gvo00003/vsc43785/Thesis/query/deployed/'
traj_dir = '/scratch/gent/vo/000/gvo00003/vsc43785/Thesis/query/300K.xyz'
results_dir = '/scratch/gent/vo/000/gvo00003/vsc43785/Thesis/query/committee_results/NVT/300K'

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
    load_mode: bool = False

    def __post_init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        logging.info('Using {} device'.format(self.device))
        assert os.path.isdir(self.models_dir), 'The models directory does not exist'
        assert os.path.isfile(self.traj_dir), 'The trajectory file does not exist'
        self.results_dir = os.path.mkdir(self.results_dir, self.name)
        if not os.path.isdir(self.results_dir):
            os.mkdir(self.results_dir)
            logging.info('Created results directory: \n{}'.format(self.results_dir))
        if not self.load_mode:
            self.store_models()
            self.load_traj()
            self.traj_e = np.zeros(self.traj_len)
            self.mean_e, self.sig_e = np.zeros(self.traj_len), np.zeros(self.traj_len)
            self.mean_f_mean, self.sig_f_mean = np.zeros(self.traj_len), np.zeros(self.traj_len)
            self.mean_f_max, self.sig_f_max = np.zeros(self.traj_len), np.zeros(self.traj_len)
            assert self.n_select<self.traj_len, 'The wanted amount of selection datapoints exceeds the amount of total datapoints: {}>{}'.format(self.n_select, self.traj_len)

    def store_models(self):
        self.models = {}
        model_file_names = [f for f in os.listdir(self.models_dir) if os.path.isfile(os.path.join(self.models_dir, f))]
        for name in model_file_names:
            if name[-4:] == '.pth':
                path = os.path.join(self.models_dir, name)
                model, metadata = load_deployed_model(path, self.device)
                assert(model.__class__.__name__ == 'RecursiveScriptModule')
                self.models[name[:-4]] = model
        self.r_max = float(metadata['r_max'])        
        self.models_len = len(self.models)
        assert self.models_len >= 2, 'The amount fo models in the committee is less than 2'
    
    def load_traj(self):
        logging.info('Loading data from ...: \n{}'.format(self.traj_dir))
        assert(self.traj_dir[-4:] == '.xyz')
        self.traj = ase.io.read(self.traj_dir, index=self.traj_index, format='extxyz')
        self.traj_len = len(self.traj)
        logging.info('... Data loaded\n')

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
            self.traj_e[i] = self.traj[i].get_total_energy()
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
        if prop == 'energy':
            disagreement = self.sig_e
        if prop == 'forces':
            if red == 'mean':
                disagreement = self.sig_f_mean
            if red == 'max':
                disagreement = self.sig_f_max 
        assert disagreement is not None, 'No valid disagreement metric was given'
        part = np.argpartition(disagreement, -self.n_select)
        ind = part[-self.n_select:]
        selected_data = self.traj[ind]
        return selected_data

    def save(self):
        logging.info('Saving results in ...: \n{}'.format(self.results_dir))
        np.save(os.path.join(self.results_dir, 'traj_e.npy'), self.traj_e)
        np.save(os.path.join(self.results_dir, 'mean_e.npy'), self.mean_e)
        np.save(os.path.join(self.results_dir, 'mean_f_mean.npy'), self.mean_f_mean)    
        np.save(os.path.join(self.results_dir, 'mean_f_max.npy'), self.mean_f_max)
        np.save(os.path.join(self.results_dir, 'sig_e.npy'), self.sig_e)
        np.save(os.path.join(self.results_dir, 'sig_f_mean.npy'), self.sig_f_mean)    
        np.save(os.path.join(self.results_dir, 'sig_f_max.npy'), self.sig_f_max)
        logging.info('... Results saved\n')
    
    def load(self):
        logging.info('Loading results from ...: \n{}'.format(self.results_dir))
        self.mean_e = np.load(os.path.join(self.results_dir, 'mean_e.npy'))
        self.mean_f_mean = np.load(os.path.join(self.results_dir, 'mean_f_mean.npy'))    
        self.mean_f_max = np.load(os.path.join(self.results_dir, 'mean_f_max.npy'))
        self.sig_e = np.load(os.path.join(self.results_dir, 'sig_e.npy'))
        self.sig_f_mean = np.load(os.path.join(self.results_dir, 'sig_f_mean.npy'))    
        self.sig_f_max = np.load(os.path.join(self.results_dir, 'sig_f_max.npy'))
        logging.info('... Results loaded\n')        

    def plot_traj_disagreement(self, from_results_dir=False):
        if from_results_dir:
            self.load()
        ylabel1 = '$E_{pred}$ [eV]'
        ylabel2 = '$\sigma_{E}$ [eV]'
        ylabel3 = '$\overline{\sigma_{\mathbf{F}}}$ [eV/$\AA$]'
        ylabel4 = '$max(\sigma_{\mathbf{F}})$ [eV/$\AA$]'
        time = np.arange(self.traj_len)

        fig, axs = plt.subplots(4, figsize=(10,12), gridspec_kw = {'wspace':0, 'hspace':0.05})
        axs[0].plot(time, self.traj_e, '.k',markersize=4, label='$MD NNP$')
        axs[0].plot(time, self.mean_e, '.', color='red',markersize=4, label='$\overline{NNPs}$')
        axs[0].set_ylabel(ylabel1)
        axs[0].set_xticklabels([])
        axs[0].legend()
        axs[0].grid()
        
        part = np.argpartition(self.sig_e, -self.n_select)
        ind = part[-self.n_select:]
        
        axs[1].plot(time, self.sig_e,'.k', markersize=4)
        axs[1].plot(time[ind],self.sig_e[ind],'.', color='red', markersize=4, label='{} highest $\sigma$'.format(self.n_select))
        axs[1].set_ylabel(ylabel2)
        axs[1].grid()
        axs[1].set_xticklabels([])
        axs[1].legend()
        
        part = np.argpartition(self.sig_f_mean, -self.n_select)
        ind = part[-self.n_select:]
        
        axs[2].plot(time, self.sig_f_mean,'.k', markersize=4)
        axs[2].plot(time[ind],self.sig_f_mean[ind],'.', color='red', markersize=4, label='{} highest $\sigma$'.format(self.n_select))
        axs[2].set_ylabel(ylabel3)
        axs[2].grid()
        axs[2].set_xticklabels([])
        axs[2].legend()
        
        part = np.argpartition(self.sig_f_max, -self.n_select)
        ind = part[-self.n_select:]
        
        axs[3].plot(time, self.sig_f_max,'.k', markersize=4)
        axs[3].plot(time[ind],self.sig_f_max[ind],'.', color='red', markersize=4, label='{} highest $\sigma$'.format(self.n_select))
        axs[3].set_ylabel(ylabel4)
        axs[3].grid()
        axs[3].set_xlabel('Step')
        axs[3].legend()
        
        plt.savefig('{}/traj_disagreement'.format(results_dir),dpi=100)

if __name__ == "__main__":
    committee = qbc(name='test', models_dir=models_dir, traj_dir=traj_dir, results_dir=results_dir, traj_index=':', n_select=100)
    committee.evaluate_committee(save=True)
    #committee.plot_traj_disagreement(from_results_dir=False)