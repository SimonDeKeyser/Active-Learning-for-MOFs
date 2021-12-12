from pathlib import Path
from qbc import qbc
import argparse

parser = argparse.ArgumentParser(
            description="Perform the last QbC evaluation."
        )
parser.add_argument("traj-dir", help="The indices of the trajectory to perform QbC on")
parser.add_argument("traj-index", help="The indices of the trajectory to perform QbC on")
args = parser.parse_args()
traj_dir = args.traj_dir
traj_index = args.traj_index

##########################################################################################

root = Path('../').resolve() 
head_dir = root / 'qbc_train'

##########################################################################################
p = head_dir.glob('**/*')
cycles = sorted([x for x in p if (x.is_dir() and x.name[:5] == 'cycle')])
eval_dir = head_dir / 'evaluation'

for cycle in cycles:

    name = 'cycle{}'.format(cycle)

    committee = qbc(name=name, models_dir=cycle / 'results', 
                        traj_dir=traj_dir, results_dir=eval_dir,
                        traj_index=traj_index, nequip_train=True
                    )

    committee.evaluate_committee(save=True)