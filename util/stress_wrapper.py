from nequip.nn import GraphModuleMixin, RescaleOutput
from nequip.train import Trainer
from nequip.model import StressForceOutput
from nequip.nn import StressOutput
import torch
import sys
if sys.version_info[1] >= 8:
    from typing import Final
else:
    from typing_extensions import Final
from pathlib import Path
from e3nn.util.jit import script
from nequip.utils import Config
from nequip.utils.versions import check_code_version, get_config_code_versions
import ase
from nequip.ase import NequIPCalculator

CONFIG_KEY: Final[str] = "config"
NEQUIP_VERSION_KEY: Final[str] = "nequip_version"
TORCH_VERSION_KEY: Final[str] = "torch_version"
E3NN_VERSION_KEY: Final[str] = "e3nn_version"
CODE_COMMITS_KEY: Final[str] = "code_commits"
R_MAX_KEY: Final[str] = "r_max"
N_SPECIES_KEY: Final[str] = "n_species"
TYPE_NAMES_KEY: Final[str] = "type_names"
JIT_BAILOUT_KEY: Final[str] = "_jit_bailout_depth"
TF32_KEY: Final[str] = "allow_tf32"

scaled_model, conf = Trainer.load_model_from_training_session('../qbc_train/cycle6/results/model3')

def wrap_model(scaled_model: RescaleOutput) -> RescaleOutput:
    gradient_model = scaled_model.model

    gradient_stress_model = StressOutput(energy_model = gradient_model.func)

    scaled_stress_model = RescaleOutput(model=gradient_stress_model, 
                                        scale_keys= scaled_model.scale_keys, 
                                        shift_keys= scaled_model.shift_keys,
                                        related_scale_keys= scaled_model.related_scale_keys,
                                        related_shift_keys= scaled_model.related_shift_keys,
                                        scale_by= scaled_model.scale_by,
                                        shift_by= scaled_model.shift_by,
                                        irreps_in= scaled_model.irreps_in
    )

    if scaled_stress_model.shift_by.nelement() == 0:
        scaled_stress_model.has_shift = None 

    if scaled_stress_model.scale_by.nelement() == 0:
        scaled_stress_model.has_scale = None 
                                        
    return scaled_stress_model

def deploy(model):
    config = Config.from_file('config.yaml', 'yaml')
    model.eval()
    if not isinstance(model, torch.jit.ScriptModule):
        model = script(model)

    metadata: dict = {}
    code_versions, code_commits = get_config_code_versions(config)
    for code, version in code_versions.items():
        metadata[code + "_version"] = version
    if len(code_commits) > 0:
        metadata[CODE_COMMITS_KEY] = ";".join(
            f"{k}={v}" for k, v in code_commits.items()
        )

    metadata[R_MAX_KEY] = str(float(config["r_max"]))
    if "allowed_species" in config:
        # This is from before the atomic number updates
        n_species = len(config["allowed_species"])
        type_names = {
            type: ase.data.chemical_symbols[atomic_num]
            for type, atomic_num in enumerate(config["allowed_species"])
        }
    else:
        # The new atomic number setup
        n_species = str(config["num_types"])
        type_names = config["type_names"]
    metadata[N_SPECIES_KEY] = str(n_species)
    metadata[TYPE_NAMES_KEY] = " ".join(type_names)

    metadata[JIT_BAILOUT_KEY] = str(config["_jit_bailout_depth"])
    metadata[TF32_KEY] = str(int(config["allow_tf32"]))
    metadata[CONFIG_KEY] = Path("config.yaml").read_text()

    metadata = {k: v.encode("ascii") for k, v in metadata.items()}
    torch.jit.save(model, 'deployed.pth', _extra_files=metadata)

def test(model=None):
    atoms = ase.io.read('../data/O_C_H_Al_F/MIL53AlF_lp.xyz', index = ':2')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calc = NequIPCalculator.from_deployed_model(
            model_path='deployed.pth',
            device=device
        )
    if model is not None:
        calc.model = script(model)
    atoms[0].calc = calc
    print(atoms[0].get_stress())
    atoms[1].calc = calc
    print(atoms[1].get_stress())

#model = wrap_model(scaled_model)
#deploy(model)
#print(model)
test()