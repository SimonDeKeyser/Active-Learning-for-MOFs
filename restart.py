from time import perf_counter
start_time = perf_counter()
import os
import logging
import argparse
import datetime as dt
from pathlib import Path

from nequip.utils import Config, load_file, atomic_write_group, finish_all_writes
from nequip.scripts.train import _set_global_options
from nequip.train.trainer_wandb import TrainerWandB
from nequip.utils.wandb import resume
from nequip.data import dataset_from_config

logging.basicConfig(format='',level=logging.INFO)

def main():
    file_name, config, walltime = parse_command_line()
    restart(file_name, config, walltime)

def parse_command_line():
    parser = argparse.ArgumentParser(
            description="Restart an existing NequIP training session."
        )
    parser.add_argument("session", help="trainer.pth from a previous training")
    parser.add_argument("--update-config", help="File containing any config parameters to update")
    parser.add_argument("--walltime", help="File containing any config parameters to update")
    args = parser.parse_args()

    file_name = args.session
    config = Config.from_file(args.update_config)

    if args.walltime is not None:
        walltime = dt.datetime.strptime(args.walltime, "%H:%M:%S").time()
        walltime = dt.timedelta(hours=walltime.hour, minutes=walltime.minute, seconds=walltime.second)
    return file_name, config, walltime

def restart(file, config, walltime = None):
    dictionary = load_file(
        supported_formats=dict(torch=["pt", "pth"]),
        filename= file + '/trainer.pth',
        enforced_format="torch",
    )

    dictionary['workdir'] = file
    dictionary['progress']['last_model_path'] = file + '/last_model.pth'
    dictionary['progress']['best_model_path'] = file + '/best_model.pth'
    dictionary['progress']['trainer_save_path'] = file + '/trainer.pth'

    dictionary.update(config)
    dictionary["run_time"] = 1 + dictionary.get("run_time", 0)
    dictionary.pop('train_idcs', None)
    dictionary.pop('val_idcs', None)

    config = Config(dictionary, exclude_keys=["state_dict", "progress"])

    _set_global_options(config)

    resume(config)
    trainer = TrainerWandB.from_dict(dictionary)

    dataset = dataset_from_config(config, prefix='dataset')
    logging.info(f"Successfully reload the data set of type {dataset}...")

    trainer.set_dataset(dataset)
    config.save(file+'/config.yaml','yaml')
    
    if not trainer._initialized:
        trainer.init()
    for callback in trainer._init_callbacks:
        callback(trainer)
    trainer.init_log()
    trainer.wall = perf_counter()

    with atomic_write_group():
        if trainer.iepoch == -1:
            trainer.save()
        if trainer.iepoch in [-1, 0]:
            trainer.save_config()

    trainer.init_metrics()

    while not trainer.stop_cond:
        trainer.epoch_step()
        trainer.end_of_epoch_save()
        if walltime is not None:
            if walltime.seconds - (perf_counter() - start_time) < 300:
                logging.info('\nEnd of walltime is nearing, breaking off training loop...\n')
                trainer.stop_arg = 'Walltime ending'
                open('finished', 'w').close()
                break
            else:
                rem = walltime.seconds - (perf_counter() - start_time)
                logging.info('{} walltime remaining'.format(dt.timedelta(seconds=rem)))
    for callback in trainer._final_callbacks:
        callback(trainer)

    trainer.final_log()

    trainer.save()
    finish_all_writes() 

    hpc_dir = (Path.cwd() / '../').resolve()
    p = hpc_dir.glob('**/*/finished')
    n_ready = len([x for x in p])
    p = hpc_dir.glob('model*')
    n_total = len([x for x in p if x.is_dir()])

    if n_total == n_ready:
        logging.info('All trainings are ready, starting next QbC cycle...')
        config = Config.from_file(str(Path.cwd() / '../../' / 'params.yaml'), 'yaml')
        runs_dir = (Path.cwd() / '../../../../' / 'QbC' / 'runs').resolve()
        next_walltime = walltime + config.walltime_per_model_add

        with open(runs_dir / 'cycle{}.sh'.format(config.cycle+1),'w') as rsh:
            rsh.write(
                '#!/bin/sh'
                '\n\n#PBS -o cycle{}_o.txt'
                '\n#PBS -e cycle{}_e.txt'
                '\n#PBS -l walltime={}'
                '\n#PBS -l nodes=1:ppn={}:gpus=1'
                '\n\nsource ~/.{}'
                '\npython ../train.py {} {}'.format(config.cycle+1, config.cycle+1, str(next_walltime), config.cores, config.env, config.cycle+1, str(next_walltime))
            )
        os.system('module swap cluster/{}; cd {}; bash cycle{}.sh'.format(config.cluster, runs_dir, config.cycle+1))

    else:
        logging.info('Only {} of {} models have finished training, waiting...'.format(n_ready, n_total))

if __name__ == "__main__":
    main()