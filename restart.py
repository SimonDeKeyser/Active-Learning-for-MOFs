from nequip.utils import Config, dataset_from_config, Output, load_file
import torch
from nequip.train.trainer_wandb import TrainerWandB
from nequip.utils.wandb import resume
import logging
logging.basicConfig(format='',level=logging.INFO)
import argparse

parser = argparse.ArgumentParser(
        description="Restart an existing NequIP training session."
    )
parser.add_argument("session", help="trainer.pth from a previous training")
parser.add_argument("--update-config", help="File containing any config parameters to update")
args = parser.parse_args()

file_name = args.session
config = Config.from_file(args.update_config)

dictionary = load_file(
    supported_formats=dict(torch=["pt", "pth"]),
    filename=file_name,
    enforced_format="torch",
)

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
print(len(trainer.dl_train))
#trainer.train()