import logging
import argparse

from nequip.utils import Config, dataset_from_config, load_file
import torch
from nequip.train.trainer_wandb import TrainerWandB
from nequip.utils.wandb import resume

logging.basicConfig(format='',level=logging.INFO)

def main():
    file_name, config = parse_command_line()
    restart(file_name, config)

def parse_command_line():
    parser = argparse.ArgumentParser(
            description="Restart an existing NequIP training session."
        )
    parser.add_argument("session", help="trainer.pth from a previous training")
    parser.add_argument("--update-config", help="File containing any config parameters to update")
    args = parser.parse_args()

    file_name = args.session
    config = Config.from_file(args.update_config)
    return file_name, config

def restart(file, config):
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

    torch.set_default_dtype(
        {"float32": torch.float32, "float64": torch.float64}[config.default_dtype]
    )
    resume(config)
    trainer = TrainerWandB.from_dict(dictionary)

    config.update(trainer.output.updated_dict())

    dataset = dataset_from_config(config)
    logging.info(f"Successfully reload the data set of type {dataset}...")

    trainer.set_dataset(dataset)
    config.save(file+'/config_final.yaml','yaml')
    trainer.train()

if __name__ == "__main__":
    main()