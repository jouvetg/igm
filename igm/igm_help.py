#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
import sys
from igm import (
    State,
    params_core,
    setup_igm_modules,
    setup_igm_params
)


def main() -> None: 
    
    parser = params_core()
    params = State()
    params.param_file = parser.get_default('param_file')
    imported_modules = setup_igm_modules(params)
    params = setup_igm_params(parser, imported_modules)
        
    # Check if no arguments were provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)  # Exit after displaying the help

if __name__ == "__main__":
    main()
