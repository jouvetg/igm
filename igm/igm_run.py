#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
import os
import igm


def main() -> None:
    state = igm.State()  # class acting as a dictionary

    parser = igm.params_core()
    params, _ = parser.parse_known_args()

    if params.logging:
        igm.add_logger(params=params, state=state, logging_level=params.logging_level)
        igm.gpu_information()
    else:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    modules_dict = igm.get_modules_list(params.param_file)
    imported_modules = igm.load_modules(modules_dict)
    imported_modules = igm.load_dependent_modules(imported_modules)

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

    # Place the computation on your device GPU ('/GPU:0') or CPU ('/CPU:0')
    with tf.device("/GPU:" + str(params.gpu_id)):
        igm.run_intializers(imported_modules, params, state)
        igm.run_processes(imported_modules, params, state)
        igm.run_finalizers(imported_modules, params, state)


if __name__ == "__main__":
    main()
