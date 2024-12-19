#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import os
import tensorflow as tf
from igm import (
    State,
    # params_core,
    save_params,
    run_intializers,
    run_processes,
    run_finalizers,
    setup_igm_modules,
    # setup_igm_params,
    print_gpu_info,
    add_logger,
    download_unzip_and_store,
    input,
    output
)

from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd  # , to_absolute_path

# @hydra.main(version_base=None)
# def my_app(_cfg: DictConfig) -> None:
# print(f"Current working directory : {os.getcwd()}")
# print(f"Orig working directory    : {get_original_cwd()}")
import hydra

OmegaConf.register_new_resolver("get_cwd", lambda x: os.getcwd())
from hydra.core.hydra_config import HydraConfig

physical_devices = tf.config.list_physical_devices("GPU")
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    state = State()  # class acting as a dictionary

    if cfg.core.gpu_info:
        print_gpu_info()

    if cfg.core.logging:
        add_logger(cfg=cfg, state=state)
        tf.get_logger().setLevel(cfg.core.logging_level)

    if cfg.core.print_params:
        print(OmegaConf.to_yaml(cfg))

    # Replacing this with `???` in the config file
    # if "input" not in cfg:
    #     raise ValueError("No input method specified. Please specify at least one.")
    
    input_methods = list(
        cfg.input.keys()
    )
    # For now, it does not seem possible to select from the defaults list (so we need to only specify one method in input)
    # print(HydraConfig.get().runtime.choices)
    


    if len(input_methods) > 1:
        raise ValueError("Only one input method is allowed")

    input_method = str(input_methods[0])
    input_module = getattr(input, input_method)
    input_module.run(cfg, state)
    
    
    output_modules = []
    if "output" in cfg:
        output_methods = list(
            cfg.output.keys()
        )
        for output_method in output_methods:
            output_module = getattr(output, output_method)
            output_modules.append(output_module)
            
        for output_module in output_modules:
            output_module.initialize(cfg, state) # do we need to initialize output modules? THis is not very clean...
        
    imported_modules = setup_igm_modules(cfg)

    if not cfg.core.url_data == "":
        download_unzip_and_store(cfg.core.url_data, cfg.core.folder_data)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.core.gpu_id)

    # Place the computation on your device GPU ('/GPU:0') or CPU ('/CPU:0')
    with tf.device(f"/GPU:{cfg.core.gpu_id}"):  # type: ignore for linting checks
        run_intializers(imported_modules, cfg, state)
        run_processes(imported_modules, cfg, state)
        run_finalizers(imported_modules, cfg, state)
        
        # Writing output files
        for output_module in output_modules:
            output_module.run(cfg, state) # do we need to initialize output modules? THis is not very clean...


if __name__ == "__main__":
    main()
