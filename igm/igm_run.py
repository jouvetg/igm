#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import os
import tensorflow as tf
from igm import (
    State,
    # params_core,
    # save_params,
    initialize_modules,
    run_model,
    # run_finalizers,
    setup_igm_modules,
    # setup_igm_params,
    print_gpu_info,
    add_logger,
    download_unzip_and_store,
    print_comp,
    input,
    output,
)

from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd  # , to_absolute_path
import hydra

OmegaConf.register_new_resolver("get_cwd", lambda x: os.getcwd())
from hydra.core.hydra_config import HydraConfig

physical_devices = tf.config.list_physical_devices("GPU")
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    state = State()  # class acting as a dictionary

    if cfg.core.hardware.gpu_info:
        print_gpu_info()

    gpus = tf.config.list_physical_devices("GPU")

    print([gpus[i] for i in cfg.core.hardware.visible_gpus])
    if gpus:
        try:
            selected_visible_gpus = [gpus[i] for i in cfg.core.hardware.visible_gpus]
            tf.config.set_visible_devices(selected_visible_gpus, "GPU")
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    if len(tf.config.list_logical_devices("GPU")) > 1:
        raise NotImplementedError(
            "Strategies for multiple GPUs are not yet implemented. Please make only one GPU visible."
        )
        # strategy = tf.distribute.MirroredStrategy()
    else:
        # if there is only one visible GPU, the id will be 0! Even when choosing a GPU that has index 4, it will only be 0 after configuring visible devices!
        # However, apply_gradients is having issues... so we have to update that first!
        # strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        strategy = tf.distribute.get_strategy()

    if cfg.core.logging:
        add_logger(cfg=cfg, state=state)
        tf.get_logger().setLevel(cfg.core.logging_level)

    if cfg.core.print_params:
        print(OmegaConf.to_yaml(cfg))

    # ! Needs to be before the input the way it is setup - otherwise, it will throw an error... (at least with local not loadncdf)
    if not cfg.core.url_data == "":
        folder_path = Path(get_original_cwd()).joinpath(cfg.core.folder_data)
        download_unzip_and_store(cfg.core.url_data, folder_path)

    imported_input_modules, imported_modules, imported_output_modules = (
        setup_igm_modules(cfg, state)
    )

    input_methods = list(cfg.input.keys())
    if len(input_methods) > 1:
        raise ValueError("Only one input method is allowed.")
    imported_input_modules[0].run(cfg, state)

    output_modules = []
    if "output" in cfg:
        output_methods = list(cfg.output.keys())
        for output_method in output_methods:
            output_module = getattr(output, output_method)
            output_modules.append(output_module)

        for output_module in output_modules:
            output_module.initialize(
                cfg, state
            )  # do we need to initialize output modules? This is not very clean...


    with strategy.scope():
        initialize_modules(imported_modules, cfg, state)
        run_model(imported_modules, imported_output_modules, cfg, state)
        # run_finalizers(imported_modules, cfg, state) # Remove finalizers?

    if cfg.core.print_comp:
        print_comp(state)


if __name__ == "__main__":
    main()
