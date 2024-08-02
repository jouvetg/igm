import os
import sys

import igm
# import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fixtures import empty_core_expected, override_core_expected

cwd = os.getcwd()
sys.path.append(cwd)

def test_empty_core():

    igm_config_dir = os.path.join(igm.__path__[0], "conf")
    with initialize_config_dir(version_base=None, config_dir=igm_config_dir, job_name="test_defaults"):
        cfg = compose(config_name="config", overrides=[f"hydra.searchpath=[file://{cwd}/param_files/empty]"])
        correct = OmegaConf.create(empty_core_expected)
        
        assert cfg == correct

def test_override_modules():

    igm_config_dir = os.path.join(igm.__path__[0], "conf")
    with initialize_config_dir(version_base=None, config_dir=igm_config_dir, job_name="test_defaults"):
        cfg = compose(config_name="config", overrides=[f"hydra.searchpath=[file://{cwd}/param_files/basic_core]"])
        print(OmegaConf.to_yaml(cfg))
        
        correct = OmegaConf.create(override_core_expected)
        assert cfg == correct
 

# if you run locally, use this...
# if __name__ == "__main__":
    
#     cwd = os.getcwd()
#     sys.path.append(cwd)

    # print(cwd)
    # test_empty_core()