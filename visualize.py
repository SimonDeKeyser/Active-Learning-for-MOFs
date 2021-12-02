from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class qbc_vis:
    def __init__(self, head_dir:Path, imgs_dir:str) -> None:
        self.head_dir = head_dir
        self.imgs_dir = head_dir / imgs_dir
        if not self.imgs_dir.exists():
            self.imgs_dir.mkdir()
        self.list_cycles()
    
    def list_cycles(self):
        p = self.head_dir.glob('**/*')
        self.cycles = sorted([x for x in p if (x.is_dir() and x.name[:5] == 'cycle')])
        self.len_cycles = len(self.cycles)
        self.cycle_names = sorted([x.name for x in self.cycles])
    
    def mean_disagreement(self, prop, red=None):
        mean_disagreements = np.zeros(self.len_cycles-1)
        i = 0
        for cycle in self.cycles:
            if cycle.name != 'cycle0':
                if prop == 'energy':
                    ylabel = '<$\sigma_{E}$> [eV]'
                    sigs = np.load(cycle / 'sig_e.npy')
                if prop == 'forces':
                    if red == 'mean':
                        ylabel = '<$\overline{\sigma_{\mathbf{F}}}$> [eV/$\AA$]'
                        sigs = np.load(cycle / 'sig_f_mean.npy')
                    if red == 'max':
                        ylabel = '<$max(\sigma_{\mathbf{F}})$> [eV/$\AA$]'
                        sigs = np.load(cycle / 'sig_f_max.npy')
                assert ylabel, 'Give a valid property: [energy, forces] and/or reduction: [mean, max]'
                mean_disagreements[i] = sigs.mean()
                i += 1
        plt.plot(self.cycle_names[:-1], mean_disagreements, '.k')
        plt.ylabel(ylabel)
        plt.savefig(self.imgs_dir / 'mean_disagreement', bbox_inches='tight')

    def epoch_metrics(self):
        for cycle in self.cycles:
            p = (cycle / 'results').glob('**/*')
            models = [x for x in p if (x.is_dir() and x.name[:5] == 'model')]
            for model in sorted(models):
                metrics_epoch = pd.read_csv(model / 'metrics_epoch.csv')
                

if __name__ == "__main__":
    head_dir = Path('/scratch/gent/vo/000/gvo00003/vsc43785/Thesis/q4/qbc_train')
    imgs_dir = 'qbc_imgs' 
    visual = qbc_vis(head_dir, imgs_dir)
    visual.mean_disagreement('forces','mean')
    #visual.epoch_metrics()
