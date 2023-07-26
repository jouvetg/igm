#!/usr/bin/env python3

"""
Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
Published under the GNU GPL (Version 3), check at the LICENSE file
"""

import os

# The (empty) class state serves in fact to use a dictionnary, but it looks like a class
class State:
    def __init__(self):
        self.__dict__ = dict()

# this create core parameters for any IGM run
def params_core():

    import argparse

    parser = argparse.ArgumentParser(description="IGM")

    parser.add_argument(
        "--working_dir",
        type=str,
        default="",
        help="Working directory (default empty string)",
    )

    return parser

# this add a logger to the state
def add_logger(params, state, logging_file="igm.log", logging_level="INFO"):
 
    import logging

    if logging_file=='':
        pathf = ''
    else:
        pathf = os.path.join(params.working_dir, logging_file)

    logging.basicConfig(
        filename=pathf,
        encoding="utf-8",
        filemode="w",
        level=getattr(logging, logging_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    state.logger = logging.getLogger("my_logger")

    if not pathf=="":
        os.system("echo rm " + pathf + " >> clean.sh" )

# Print parameters in screen and a dedicated file
def print_params(params):

    with open(os.path.join(params.working_dir, "igm-run-parameters.txt"), "w") as f:
        print("PARAMETERS ARE ...... ")
        for ck in params.__dict__:
            print("%30s : %s" % (ck, params.__dict__[ck]))
            print("%30s : %s" % (ck, params.__dict__[ck]), file=f)

    os.system("echo rm " + "igm-run-parameters.txt" + " >> clean.sh")
