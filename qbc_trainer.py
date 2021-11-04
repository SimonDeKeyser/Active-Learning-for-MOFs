import os
import logging
logging.basicConfig(format='',level=logging.INFO)

from qbc import qbc

nequip_train_dir = '/scratch/gent/vo/000/gvo00003/vsc43785/Thesis/query/results'
traj_dir = '/scratch/gent/vo/000/gvo00003/vsc43785/Thesis/query/300K.xyz'
results_dir = '/scratch/gent/vo/000/gvo00003/vsc43785/Thesis/query/committee_train'
cycle = 1

logging.info('### Query by Commitee ###\n')

nequip_train_dir = '/scratch/gent/vo/000/gvo00003/vsc43785/Thesis/query/results'

logging.info('Cycle {}:'.format(cycle))
name = 'cycle_{}'.format(cycle)
committee = qbc(name=name, models_dir=nequip_train_dir, traj_dir=traj_dir, results_dir=results_dir, traj_index=':10', n_select=2, nequip_train=True)
committee.evaluate_committee(save=True)
committee.plot_traj_disagreement()