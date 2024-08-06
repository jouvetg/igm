import os
import sys

import igm
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fixtures import empty_core_expected, override_core_expected, override_core_expected_cli

cwd = os.getcwd()
sys.path.append(cwd)

def test_empty_core():

    igm_config_dir = os.path.join(igm.__path__[0], "conf")
    with initialize_config_dir(version_base=None, config_dir=igm_config_dir, job_name="test_defaults"):
        cfg = compose(config_name="config", overrides=[f"hydra.searchpath=[file://{cwd}]", "+experiment=empty"])
        correct = OmegaConf.create(empty_core_expected)
        
        assert cfg == correct

def test_override_modules():

    igm_config_dir = os.path.join(igm.__path__[0], "conf")
    with initialize_config_dir(version_base=None, config_dir=igm_config_dir, job_name="test_defaults"):
        cfg = compose(config_name="config", overrides=[f"hydra.searchpath=[file://{cwd}]", "+experiment=override"])
        correct = OmegaConf.create(override_core_expected)
        
        assert cfg.core == correct.core
        assert cfg.modules == correct.modules
        assert cfg == correct

def test_comments():

    igm_config_dir = os.path.join(igm.__path__[0], "conf")
    with initialize_config_dir(version_base=None, config_dir=igm_config_dir, job_name="test_defaults"):
        cfg = compose(config_name="config", overrides=[f"hydra.searchpath=[file://{cwd}]", "+experiment=override_with_comments"])
        correct = OmegaConf.create(override_core_expected)
        
        assert cfg.core == correct.core
        assert cfg.modules == correct.modules
        assert cfg == correct

def test_command_line_override():

    igm_config_dir = os.path.join(igm.__path__[0], "conf")
    with initialize_config_dir(version_base=None, config_dir=igm_config_dir, job_name="test_defaults"):
        cfg = compose(config_name="config", overrides=[f"hydra.searchpath=[file://{cwd}]",
                                                       "+experiment=override_with_comments",
                                                        "core.saved_params_filename=overriden_params_cli",
                                                        "modules.iceflow.iceflow.iflo_fieldin=[thk, usurf]",
                                                        "core.gpu_id=3"])
        
        correct = OmegaConf.create(override_core_expected_cli)
        
        assert cfg.core == correct.core
        assert cfg.modules == correct.modules
        assert cfg == correct
        
def test_incorrect_module_name():

    igm_config_dir = os.path.join(igm.__path__[0], "conf")
    with initialize_config_dir(version_base=None, config_dir=igm_config_dir, job_name="test_defaults"):
        
        with pytest.raises(ModuleNotFoundError):
            cfg = compose(config_name="config", overrides=[f"hydra.searchpath=[file://{cwd}]", "+experiment=incorrect_module_name"])
        
        # correct = OmegaConf.create(override_core_expected_cli)
        
        # assert cfg.core == correct.core
        # assert cfg.modules == correct.modules
        # assert cfg == correct
 

# if you run locally, use this...
# if __name__ == "__main__":
    
#     cwd = os.getcwd()
#     sys.path.append(cwd)

    # print(cwd)
    # test_empty_core()