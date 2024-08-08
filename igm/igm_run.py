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
    download_unzip_and_store
)

from omegaconf import DictConfig, OmegaConf
import hydra

OmegaConf.register_new_resolver("get_cwd", lambda x: os.getcwd())

physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)
    
# param_filename = OmegaConf.load("params.yaml")
# OmegaConf.register_new_resolver("get_param_file", lambda x: "thk")


@hydra.main(version_base=None, config_path="conf", config_name="config")
# @hydra.main(version_base=None, config_path=".", config_name="params")
def main(cfg: DictConfig) -> None:
    state = State()  # class acting as a dictionary
    # parser = params_core()
    # params, _ = parser.parse_known_args()
    # cfg
    
    # if cfg.search_path is not None:
    #     override_path = hydra.utils.to_absolute_path(cfg.search_path)
    #     # print(override_path)
    #     override_conf = OmegaConf.load(override_path)
    #     # print('original cfg', OmegaConf.to_yaml(cfg))
    #     cfg = OmegaConf.merge(cfg, override_conf)
    
    # override_path = hydra.utils.to_absolute_path(f"{cfg.cwd}/params.yaml")
    # print(override_path)
    # override_conf = OmegaConf.load(override_path)
    # print('original cfg', OmegaConf.to_yaml(cfg))
    # cfg = OmegaConf.merge(cfg, override_conf)
        # print('merged cfg', OmegaConf.to_yaml(cfg))
    
    # print("yee",cfg.search_path)

    # print(OmegaConf.to_yaml(cfg))
    # print(cfg.testd)
    # import time
    # time.sleep(15)
    # exit()

    if cfg.core.gpu_info:
        print_gpu_info()

    if cfg.core.logging:
        add_logger(cfg=cfg, state=state)
        tf.get_logger().setLevel(cfg.logging_level)
        
    imported_modules = setup_igm_modules(cfg)
    
    # params = setup_igm_params(parser, imported_modules)
    # print(OmegaConf.to_yaml(cfg.modules.iceflow.iceflow))
    # print(imported_modules)
    # exit()
    if cfg.core.print_params:
        print(OmegaConf.to_yaml(cfg))
        # save_params(cfg) # already handled with logging it seems... (hydra specifically)
        
    if not cfg.core.url_data=="":
        download_unzip_and_store(cfg.core.url_data, cfg.core.folder_data)
        
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.core.gpu_id)        

    # print(imported_modules)
    
    # Place the computation on your device GPU ('/GPU:0') or CPU ('/CPU:0')
    
    # exit()
    with tf.device(f"/GPU:{cfg.core.gpu_id}"):  # type: ignore for linting checks
        run_intializers(imported_modules, cfg, state)
        run_processes(imported_modules, cfg, state)
        run_finalizers(imported_modules, cfg, state)


if __name__ == "__main__":
    main()
