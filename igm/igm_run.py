#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
import os
import igm
from typing import List, Any
import logging
import json


def run_intializers(modules: List, params: Any, state: igm.State) -> None:
    for module in modules:
        module.initialize(params, state)


def run_processes(modules: List, params: Any, state: igm.State) -> None:
    if hasattr(state, "t"):
        while state.t < params.time_end:
            # with tf.profiler.experimental.Trace(
            #     "process_profile", step_num=state.t, _r=1
            # ):
            for module in modules:
                module.update(params, state)


def run_finalizers(modules: List, params: Any, state: igm.State) -> None:
    for module in modules:
        module.finalize(params, state)


def gpu_information():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    logging.info(f"{'CUDA Enviroment':-^150}")
    logging.info(f"{json.dumps(tf.sysconfig.get_build_info(), indent=2, default=str)}")
    logging.info(f"{'Available GPU Devices':-^150}")
    for gpu in gpus:
        logging.info(f" Name: {gpu.name} Type: {gpu.device_type}")
        logging.info(
            f"{json.dumps(tf.config.experimental.get_device_details(gpu), indent=2, default=str)}"
        )
    logging.info(f"{'':-^150}")


def main():
    # Collect defaults, overide from json file, and parse all core parameters
    parser = igm.params_core()
    params, _ = parser.parse_known_args()

    modules_dict = igm.get_modules_list(params.param_file)
    imported_modules = igm.load_modules(modules_dict)
    imported_modules = igm.load_dependent_modules(imported_modules)

    # Collect defaults, overide from json file, and parse all specific module parameters
    for module in imported_modules:
        module.params(parser)

    core_and_module_params = parser.parse_args()
    params = igm.load_user_defined_params(
        param_file=core_and_module_params.param_file,
        params_dict=vars(core_and_module_params),
    )

    parser.set_defaults(**params)
    params = parser.parse_args()

    if params.print_params:
        igm.print_params(params)

    # Define a state class/dictionnary that contains all the data
    state = igm.State()

    # if logging is activated, add a logger to the state
    if params.logging:
        igm.add_logger(params, state)
        gpu_information()
    else:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Place the computation on your device GPU ('/GPU:0') or CPU ('/CPU:0')
    with tf.device("/GPU:" + str(params.gpu_id)):
        # Initialize all the model components in turn
        run_intializers(imported_modules, params, state)
        run_processes(imported_modules, params, state)
        run_finalizers(imported_modules, params, state)


if __name__ == "__main__":
    main()
