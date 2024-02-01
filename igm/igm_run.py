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
    setup_igm,
)


def main() -> None:
    state = State()  # class acting as a dictionary
    parser = params_core()
    imported_modules, params, state = setup_igm(state=state, parser=parser)

    if params.print_params:
        print_params(params=params)

    # Place the computation on your device GPU ('/GPU:0') or CPU ('/CPU:0')
    with tf.device(f"/GPU:{params.gpu_id}"):  # type: ignore for linting checks
        run_intializers(imported_modules, params, state)
        run_processes(imported_modules, params, state)
        run_finalizers(imported_modules, params, state)


if __name__ == "__main__":
    main()
