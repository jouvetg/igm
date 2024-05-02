#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from igm import (
    State,
    params_core,
    print_params,
    run_intializers,
    run_processes,
    run_finalizers,
    setup_igm_modules,
    setup_igm_params,
    print_gpu_info,
    add_logger,
    download_unzip_and_store
)


def main() -> None:
    state = State()  # class acting as a dictionary
    parser = params_core()
    params, _ = parser.parse_known_args()

    if params.gpu_info:
        print_gpu_info()

    if params.logging:
        add_logger(params=params, state=state)
        tf.get_logger().setLevel(params.logging_level)
        
    imported_modules = setup_igm_modules(params)
    params = setup_igm_params(parser, imported_modules)

    if params.print_params:
        print_params(params=params)
        
    if not params.url_data=="":
        download_unzip_and_store(params.url_data,params.folder_data)

    # Place the computation on your device GPU ('/GPU:0') or CPU ('/CPU:0')
    with tf.device(f"/GPU:{params.gpu_id}"):  # type: ignore for linting checks
        run_intializers(imported_modules, params, state)
        run_processes(imported_modules, params, state)
        run_finalizers(imported_modules, params, state)


if __name__ == "__main__":
    main()
