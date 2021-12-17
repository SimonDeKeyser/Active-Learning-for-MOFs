from os import read
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from natsort import natsorted


class qbc_vis:
    def __init__(self, head_dir:Path, imgs_dir:str, eval_dir:str = 'evaluation') -> None:
        self.head_dir = head_dir
        self.imgs_dir = head_dir / imgs_dir
        if not self.imgs_dir.exists():
            self.imgs_dir.mkdir()
        self.list_cycles()
        self.eval_dir = self.head_dir / eval_dir
    
    def list_cycles(self):
        p = self.head_dir.glob('*')
        self.cycle_names = natsorted([x.name for x in p if (x.is_dir() and x.name[:5] == 'cycle')])
        self.cycles = [(self.head_dir / name) for name in self.cycle_names]
        self.len_cycles = len(self.cycles)
    
    def mean_disagreement(self, prop, red=None, combine=False):
        mean_disagreements = np.zeros(self.len_cycles-1)
        i = 0
        for cycle in self.cycles:
            if cycle.name != 'cycle0':
                if prop == 'energy':
                    ylabel = '<$\sigma_{E}$> [meV]'
                    sigs = np.load(cycle / 'sig_e.npy')
                if prop == 'forces':
                    if red == 'mean':
                        ylabel = '<$\overline{\sigma_{\mathbf{F}}}$> [meV/$\AA$]'
                        sigs = np.load(cycle / 'sig_f_mean.npy')
                    if red == 'max':
                        ylabel = '<$max(\sigma_{\mathbf{F}})$> [meV/$\AA$]'
                        sigs = np.load(cycle / 'sig_f_max.npy')
                assert ylabel, 'Give a valid property: [energy, forces] and/or reduction: [mean, max]'
                mean_disagreements[i] = sigs.mean()
                i += 1

        if combine:
            return 1000*mean_disagreements
        else:
            plt.plot(np.arange(self.len_cycles-1), 1000*mean_disagreements, '.k')
            plt.ylabel(ylabel)
            plt.savefig(self.imgs_dir / 'mean_disagreement', bbox_inches='tight')
            plt.close()

    def best_mae_hour(self, metrics):
        clean_df = metrics[metrics[' wall'] != ' wall']
        cycle_stops = metrics[metrics[' wall'] == ' wall'].index
        cycle_stops = [-1] + list(cycle_stops)
        wall = []
        last_cycle_wall = 0
        cycle_walls = []
        for i in range(len(cycle_stops)-1):
            wall += list(metrics[' wall'].iloc[cycle_stops[i]+1:cycle_stops[i+1]].to_numpy(np.float32) + last_cycle_wall)
            last_cycle_wall = wall[-1]
            cycle_walls.append(last_cycle_wall/3600)
        wall += list(metrics[' wall'].iloc[cycle_stops[-1]+1:].to_numpy(np.float32) + last_cycle_wall)
        best_mae = []
        time = []
        for h in range(1,48):
            if not h > wall[-1]/3600:
                best_mae.append(clean_df[np.array(wall)/3600 < h][' Validation_all_f_mae'].to_numpy(dtype=np.float64).min())
                time.append(h)
            else:
                break
        return np.array(time), np.array(best_mae), np.array(cycle_walls)

    def train_mae_hour(self, metrics):
        clean_df = metrics[metrics[' wall'] != ' wall']
        cycle_stops = metrics[metrics[' wall'] == ' wall'].index
        cycle_stops = [-1] + list(cycle_stops)
        wall = []
        last_cycle_wall = 0
        cycle_walls = []
        for i in range(len(cycle_stops)-1):
            wall += list(metrics[' wall'].iloc[cycle_stops[i]+1:cycle_stops[i+1]].to_numpy(np.float32) + last_cycle_wall)
            last_cycle_wall = wall[-1]
            cycle_walls.append(last_cycle_wall/3600)
        wall += list(metrics[' wall'].iloc[cycle_stops[-1]+1:].to_numpy(np.float32) + last_cycle_wall)
        train_mae = []
        time = []
        for h in np.arange(0.1,48,0.25):
            if not h > wall[-1]/3600:
                train_mae.append(clean_df[np.array(wall)/3600 < h][' Training_all_f_mae'].to_numpy(dtype=np.float64)[-1])
                time.append(h)
            else:
                break
        return np.array(time), np.array(train_mae), np.array(cycle_walls)

    def epoch_metrics(self, c=-1):
        p = (self.cycles[c] / 'results').glob('*')
        models = [x for x in p if (x.is_dir() and x.name[:5] == 'model')]
        mx = 0
        mn = 10
        fig, ax = plt.subplots()
        for model in sorted(models):
            metrics_epoch = pd.read_csv(model / 'metrics_epoch.csv')
            time, best_mae, cycle_stops = self.best_mae_hour(metrics_epoch)
            if max(best_mae) > mx:
                mx = max(best_mae)
            if min(best_mae) < mn:
                mn = min(best_mae)
            ax.plot(time, 1000*best_mae, '.--',label=model.name)
        count = 1
        for cycle_stop in cycle_stops:
            ax.vlines(cycle_stop, 1000*mn, 1000*mx,color='black')
            ax.text(cycle_stop + 0.1, 1000*mn, str(count))
            count += 1
        ax.set_ylabel('Validation Forces Best MAE [meV/$\AA$]')
        ax.set_xlabel('Training Time [h]')
        ax.set_yscale('log')
        ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        plt.legend()
        plt.savefig(self.imgs_dir / 'val_f_mae', bbox_inches='tight')
        plt.close()

        mx = 0
        mn = 10
        fig, ax = plt.subplots()
        for model in sorted(models):
            metrics_epoch = pd.read_csv(model / 'metrics_epoch.csv')
            time, train_mae, cycle_stops = self.train_mae_hour(metrics_epoch)
            if max(train_mae) > mx:
                mx = max(train_mae)
            if min(train_mae) < mn:
                mn = min(train_mae)
            ax.plot(time, 1000*train_mae, '.--',label=model.name)
        count = 1
        for cycle_stop in cycle_stops:
            ax.vlines(cycle_stop, 1000*mn, 1000*mx,color='black')
            ax.text(cycle_stop + 0.1, 1000*mn, str(count))
            count += 1
        ax.set_ylabel('Training Forces MAE [meV/$\AA$]')
        ax.set_xlabel('Training Time [h]')
        ax.set_yscale('log')
        ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        plt.legend()
        plt.savefig(self.imgs_dir / 'train_f_mae', bbox_inches='tight')
        plt.close()

    def evaluation(self, combine = False):
        assert self.eval_dir.exists(), 'The evaluation directory does not exist, do an evaluation first'

        p = (self.eval_dir / 'cycle0').glob('*')
        model_files = [x for x in p if (x.is_dir() and x.name[:5] == 'model')]
        model_names = [x.name for x in model_files]

        all_f_mae = {}
        e_mae = {}
        for model in sorted(model_names):
            all_f_mae[model] = []
            e_mae[model] = [] 
        for cycle in self.cycle_names:
            for model in sorted(model_names):
                f = open(self.eval_dir / cycle / model / 'output', 'r')
                lines = f.readlines()
                for line in lines:
                    if '=' in line:
                        metric, value = line.split('=')
                        metric = metric.strip()
                        if metric == 'all_f_mae':
                            all_f_mae[model].append(1000*float(value))
                        if metric == 'e_mae':
                            e_mae[model].append(1000*float(value))
        
        if combine:
            best = ''
            mn = 100
            for key in all_f_mae.keys():
                if all_f_mae[key][-1] <= mn:
                    best = key
                    mn = all_f_mae[key][-1]
            return all_f_mae[best]
        else:
            for key in all_f_mae.keys():
                plt.plot(self.cycle_names,all_f_mae[key],'.--',label=key)
            plt.xlabel('QbC cycle')
            plt.ylabel('Forces MAE (test set) [meV/$\AA$]')
            plt.legend()
            plt.savefig(self.imgs_dir / 'test_f_mae', bbox_inches='tight')
            plt.close()

def combine_disagreement(train_folders, labels, prop, red=None):
    if prop == 'energy':
        ylabel = '<$\sigma_{E}$> [meV]'    
    if prop == 'forces':
        if red == 'mean':
            ylabel = '<$\overline{\sigma_{\mathbf{F}}}$> [meV/$\AA$]'
        if red == 'max':
            ylabel = '<$max(\sigma_{\mathbf{F}})$> [meV/$\AA$]'

    thesis_dir = Path('../../').resolve()
    for i, folder in enumerate(train_folders):
        vis = qbc_vis(thesis_dir / folder, imgs_dir)
        disagreement = vis.mean_disagreement(prop, red, combine=True)
        plt.plot(np.arange(vis.len_cycles - 1), disagreement,'.--',label=labels[i])
    plt.ylabel(ylabel)
    plt.xlabel('QbC cycle')
    plt.legend()
    plt.savefig(vis.imgs_dir / 'total_{}_{}_disagreement'.format(prop, red), bbox_inches='tight')
    plt.close()

def combine_test(eval_dir, train_folders, labels):
    thesis_dir = Path('../../').resolve()
    for i, folder in enumerate(train_folders):
        vis = qbc_vis(thesis_dir / folder, imgs_dir, eval_dir)
        all_f_mae = vis.evaluation(combine=True)
        plt.plot(all_f_mae,'.--',label=labels[i])
    plt.ylabel('Forces MAE (test set) [meV/$\AA$]')
    plt.xlabel('QbC cycle')
    plt.legend()
    plt.savefig(vis.imgs_dir / 'total_test', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    head_dir = Path('../').resolve() / 'qbc_train'
    imgs_dir = 'qbc_imgs' 
    visual = qbc_vis(head_dir, imgs_dir)
    #visual.mean_disagreement('forces','mean')
    #visual.epoch_metrics()
    #visual.evaluation()

    #plot_test('evaluation')

    train_folders = ['q4_mean10/qbc_train', 'q4_random10/qbc_train']
    labels = ['mean', 'random']
    prop = 'forces'
    red = 'mean'
    combine_disagreement(train_folders, labels, prop, red)

    train_folders = ['q4/qbc_train', 'q4_max/qbc_train', 'q4_random/qbc_train']
    labels = ['mean','max', 'random']
    prop = 'forces'
    red = 'mean'
    combine_disagreement(train_folders, labels, prop, red)

    train_folders = ['q4/qbc_train', 'q4_max/qbc_train', 'q4_random/qbc_train']
    labels = ['mean','max', 'random']
    prop = 'forces'
    red = 'max'
    combine_disagreement(train_folders, labels, prop, red)