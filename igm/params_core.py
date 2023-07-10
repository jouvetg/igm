#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file 

import argparse


def params_core():
    # this create core parameters
    parser = argparse.ArgumentParser(description="IGM")

    parser.add_argument(
        "--working_dir",
        type=str,
        default="",
        help="Working directory (default empty string)",
    )
    parser.add_argument(
        "--logging_level", type=str, default="CRITICAL", help="Looging level"
    )
    parser.add_argument(
        "--logging_file", type=str, default="igm.log", help="Looging level"
    )

    return parser