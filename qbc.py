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
from nequip.data import AtomicData, dataset_from_config, Collater
from nequip.train import Trainer

@dataclass
class CNNP:
    """
    Class which contains the Committee Neural Network Potential
    """
    models_dir: str

    def __post_init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        logging.info('Using {} device'.format(self.device))
        assert self.models_dir.is_dir(), 'The models directory does not exist'

    def load_models_from_nequip_training(self):
        self.models = {}
        p = Path(self.models_dir).glob('*')
        model_files = [x for x in p if x.is_dir()]
        logging.info('Models found in the NequIP training directory:')
        for file in sorted(model_files):
            if 'model' in file.name:
                logging.info('*\t{}'.format(file.name))
                model, self.config = Trainer.load_model_from_training_session(traindir=file)
                model = model.to(self.device)
                model.eval()
                self.models[file.name] = model  
    
    def load_models_from_nequip_deployed(self):
        self.models = {}
        p = Path(self.models_dir).glob('*')
        model_files = [x for x in p if x.is_file()]
        logging.info('Models found in the models directory:')
        for file in model_files:
            if file.suffix == '.pth':
                logging.info('*\t{}'.format(file.name[:-4]))
                model, _ = load_deployed_model(file, self.device)
                assert(model.__class__.__name__ == 'RecursiveScriptModule')
                self.models[file.name[:-4]] = model

    def __len__(self):
        if self.models is None:
            return 0
        else:
            return len(self.models) 

    def forward(self, batch):
        batch = batch.to(self.device)
        energies = np.zeros((len(self), self.batch_size))
        forces = np.zeros((len(self), batch['forces'].shape[0], 3))

        results = [model(AtomicData.to_AtomicDataDict(batch)) for model in self.models.values()]
        energies = torch.stack([result['total_energy'] for result in results], dim=-1)
        forces = torch.stack([result['forces'] for result in results], dim=-1)

        return energies, forces

@dataclass
class QbC:
    """
    Class which performs the Query by Committee
    """
    name: str
    cnnp: CNNP
    traj_dir: str
    results_dir: str
    traj_index: str = ':'
    n_select: int = 100
    r_max: float = 4.5
    batch_size: int = 5

    def __post_init__(self):
        self.device = self.cnnp.device
        assert self.traj_dir.exists(), 'The trajectory file does not exist'
        self.results_dir = self.results_dir / self.name
        if not self.results_dir.exists():
            self.results_dir.mkdir()
            logging.info('Created results directory: \n{}'.format(self.results_dir))
        self.load_traj()
        self.traj_e = np.zeros(self.traj_len)
        self.mean_e, self.sig_e = np.zeros(self.traj_len), np.zeros(self.traj_len)
        self.mean_f_mean, self.sig_f_mean = np.zeros(self.traj_len), np.zeros(self.traj_len)
        self.mean_f_max, self.sig_f_max = np.zeros(self.traj_len), np.zeros(self.traj_len)
        assert self.n_select<self.traj_len, 'The wanted amount of selection datapoints exceeds the amount of total datapoints: {}>{}'.format(self.n_select, self.traj_len)
    
    def load_traj(self):
        logging.info('Loading trajectory from ...: \n{}'.format(self.traj_dir))
        assert(self.traj_dir.suffix == '.xyz')
        self.cnnp.config.dataset_file_name = str(self.traj_dir)
        self.cnnp.config.ase_args['index'] = self.traj_index
        self.cnnp.config.root = str(self.results_dir)
        self.traj = dataset_from_config(self.cnnp.config, prefix='dataset')
        self.traj_len = len(self.traj)
        logging.info('... Trajectory loaded\n')

    def evaluate_committee(self, save=False):
        logging.info('Starting evaluation ...')
        logging.info('Batch:')
        c = Collater.for_dataset(self.traj, exclude_keys=[])
        test_idcs = np.arange(self.traj_len)
        batch_i = 0
        while True:
            this_batch_test_indexes = test_idcs[
                batch_i * self.batch_size : (batch_i + 1) * self.batch_size
            ]
            datas = [self.traj[int(idex)] for idex in this_batch_test_indexes]
        
            if len(datas) == 0:
                break
            n_atoms = np.array([data.positions.shape[0] for data in datas])
            batch = c.collate(datas)
            batch = batch.to(self.device)

            self.traj_e[this_batch_test_indexes] = batch['total_energy'].cpu().numpy().flatten() 
            energies, forces = self.cnnp.forward(batch)

            energies = energies.detach().cpu().numpy()
            forces = forces.detach().cpu().numpy()
            
            self.mean_e[this_batch_test_indexes] = energies.mean(-1).flatten()
            self.sig_e[this_batch_test_indexes] = energies.std(-1).flatten()
            
            self.mean_f_mean[this_batch_test_indexes] = np.array([abs(forces.mean(-1))[n_atoms[:b].sum(): n_atoms[:b].sum() + n_atoms[b]].mean() for b in range(self.batch_size)])
            self.sig_f_mean[this_batch_test_indexes] = np.array([forces.std(-1)[n_atoms[:b].sum(): n_atoms[:b].sum() + n_atoms[b]].mean() for b in range(self.batch_size)])
            
            self.mean_f_max[this_batch_test_indexes] = np.array([abs(forces.mean(-1))[n_atoms[:b].sum(): n_atoms[:b].sum() + n_atoms[b]].max() for b in range(self.batch_size)])
            self.sig_f_max[this_batch_test_indexes] = np.array([forces.std(-1)[n_atoms[:b].sum(): n_atoms[:b].sum() + n_atoms[b]].max() for b in range(self.batch_size)])
            logging.info('[{}/{}]'.format(batch_i,self.traj_len//self.batch_size))
            batch_i += 1
            
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
        selected_data = [self.traj[i].to(device="cpu").to_ase(
                                type_mapper=self.traj.type_mapper)
                        for i in ind]
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