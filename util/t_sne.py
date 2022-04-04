from nequip.train import Trainer
from nequip.data import AtomicData, dataset_from_config
from pathlib import Path
import ase.io 
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from cycler import cycler
plt.style.use('default')
plt.rcParams['axes.prop_cycle'] = cycler(marker= ['o', 'D', 'X']) * cycler(color= ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']) 
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
plt.rcParams['mathtext.fontset'] = 'dejavusans'

def forward(path_atoms):
    conf.dataset_file_name = str(path_atoms)
    conf.ase_args['index'] = ':'
    conf.root = str(Path.cwd() / 'datasets')
    atoms = dataset_from_config(conf, prefix='dataset')
    ase_atoms = ase.io.read(path_atoms, index=0, format='extxyz')
    inds = ase_atoms.get_atomic_numbers() == 6
    dfs = []
    for i in range(len(atoms)):
        _ = model(AtomicData.to_AtomicDataDict(atoms[i]))
        feats = features['feats'][inds]
        df2 = pd.DataFrame(columns=[str(i) for i in range(16)], data = feats)
        df1 = pd.DataFrame(columns=['MOF'], data = [path_atoms.name[:-4] for i in range(feats.shape[0])])
        dfs.append(pd.concat([df1, df2], axis=1))
    return pd.concat(dfs)

def cp2k(path_atoms):
    atoms = ase.io.read(path_atoms, index=':', format='extxyz')
    inds = atoms[0].get_atomic_numbers() == 6
    dfs = []
    for i in range(len(atoms)):
        feats = np.concatenate((atoms[i].positions[inds], atoms[i].arrays['forces'][inds]), axis=1)
        df2 = pd.DataFrame(columns=[str(i) for i in range(16)], data = feats)
        df1 = pd.DataFrame(columns=['MOF'], data = [path_atoms.name[:-4] for i in range(feats.shape[0])])
        dfs.append(pd.concat([df1, df2], axis=1))
    return pd.concat(dfs)

def create_df():
    dataframes = []
    for f in files:
        df = cp2k(f)
        dataframes.append(df)

    total_df = pd.concat(dataframes)
    print(total_df)
    total_df.to_csv('t_sne.csv')
    return total_df

load= False
load_emb = False

p = (Path.cwd() / '../data').glob('**/*.xyz')
unsort_files = [x for x in p]
names = [f.name for f in unsort_files]
files = []
for name in sorted(names):
    for f in unsort_files:
        if f.name == name:
            files.append(f)

mapping = {}
count = 0
for f in files:
    mapping[f.name[:-4]] = count
    count += 1

if not load_emb:
    if load:
        df = pd.read_csv('t_sne.csv')
    else:
        path_model = Path('/kyukon/scratch/gent/vo/000/gvo00003/vsc43785/Thesis/q4_MOFs/qbc_train/cycle6/results/model3')
        model, conf = Trainer.load_model_from_training_session(path_model)

        features = {}
        def get_features(name):
            def hook(model, input, output):
                features[name] = output.detach().numpy()
            return hook

        model.model.func.conv_to_output_hidden.linear.register_forward_hook(get_features('feats'))
        df = create_df()

    reducer = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity= 50)
    data = df[[str(i) for i in range(16)]].values
    embedding = reducer.fit_transform(data)
    np.save('t_sne.npy', embedding)
else:
    df = pd.read_csv('t_sne.csv')
    embedding = np.load('t_sne.npy')

labels = [f.name[:-4] for f in files]
plt.scatter(embedding[:, 0], embedding[:, 1], c = df.MOF.map(mapping), cmap='tab20', s= 5)
plt.gca().set_aspect('equal', 'datalim')
cbar = plt.colorbar(boundaries=np.arange(len(files)+1)-0.5)
cbar.set_ticks(np.arange(len(files)))
cbar.set_ticklabels(labels)
plt.xlabel('Reduced dimension 1 [a.u.]')
plt.ylabel('Reduced dimension 2 [a.u.]')
plt.savefig('t_sne.png', bbox_inches='tight')
