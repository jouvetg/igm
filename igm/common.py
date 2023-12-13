#!/usr/bin/env python3

"""
Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
Published under the GNU GPL (Version 3), check at the LICENSE file
"""

import os, glob, json, sys, inspect, re
import importlib
import argparse
from igm.modules.utils import *
import igm


# The (empty) class state serves in fact to use a dictionnary, but it looks like a class
class State:
    def __init__(self):
        self.__dict__ = dict()


# this create core parameters for any IGM run
def params_core():
    parser = argparse.ArgumentParser(description="IGM")

    parser.add_argument(
        "--working_dir",
        type=str,
        default="",
        help="Working directory (default empty string)",
    )
    parser.add_argument(
        "--modules_preproc",
        type=list,
        default=["oggm_shop"],
        help="List of pre-processing modules",
    )
    parser.add_argument(
        "--modules_process",
        type=list,
        default=["iceflow", "time", "thk"],
        help="List of processing modules",
    )
    parser.add_argument(
        "--modules_postproc",
        type=list,
        default=["write_ncdf", "plot2d", "print_info"],
        help="List of post-processing modules",
    )
    parser.add_argument(
        "--logging",
        type=str2bool,
        default=False,
        help="Activate the looging",
    )
    parser.add_argument(
        "--logging_file",
        type=str,
        default="",
        help="Logging file name, if empty it prints in the screen",
    )
    parser.add_argument(
        "--print_params",
        type=str,
        default=True,
        help="Print definitive parameters in a file for record",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Id of the GPU to use (default 0)",
        
    )
    return parser


# Function to remove comments from a JSON string
def remove_comments(json_str):
    lines = json_str.split("\n")
    cleaned_lines = [line for line in lines if not line.strip().startswith("#")]
    return "\n".join(cleaned_lines)


def overide_from_json_file(parser, check_if_params_exist=True):
    # get the path of the json file
    param_file = os.path.join(parser.parse_args(args=[]).working_dir, "params.json")

    # load the given parameters from the json file
    with open(param_file, "r") as json_file:
        json_text = json_file.read()

    # Remove comments from the JSON string
    json_without_comments = remove_comments(json_text)

    # Parse the modified JSON string
    try:
        dic_params = json.loads(json_without_comments)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

    # list only the parameters registered so far
    LIST = list(vars(parser.parse_args(args=[])).keys())

    if "time_step" in dic_params["modules_process"]:
        import sys

        print(
            " ------- CHECK THE WIKI TO CHANGE YOUR PARAM FILE AND USER MODULES -------------"
        )
        sys.exit(
            "WARNING: the parameter time_step is deprecated, UPDATE you params.json file"
        )

    if check_if_params_exist:
        for key in dic_params.keys():
            if not key in LIST:
                print(
                    "WARNING: the following parameters of the json file do not exist in igm: ",
                    key,
                )

    # keep only the parameters to overide hat were registerd so far
    filtered_dict = {key: value for key, value in dic_params.items() if key in LIST}

    parser.set_defaults(**filtered_dict)


# this add a logger to the state
def load_custom_module(params, module):
    if os.path.exists(os.path.join(params.working_dir, module + ".py")):
        sys.path.append(params.working_dir)
        custmod = importlib.import_module(module)

        def is_user_defined_function(obj):
            return inspect.isfunction(obj) and inspect.getmodule(obj) == custmod

        LIST = [
            name
            for name, obj in inspect.getmembers(custmod)
            if is_user_defined_function(obj)
        ]

        for st in LIST:
            vars(igm)[st] = vars(custmod)[st]


# this function checks if there dependency modules that are not in the module list already
# dependent modules are listed by function dependency_MODULENAME, and exist only if this function exists.
def find_dependent_modules(modules):
    dm = sum(
        [
            getattr(igm, "dependency_" + m)()
            for m in modules
            if hasattr(igm, "dependency_" + m)
        ],
        [],
    )

    return [m for m in dm if m not in modules]


# this add a logger to the state
def add_logger(params, state, logging_level="INFO"):
    import logging

    if params.logging_file == "":
        pathf = ""
    else:
        pathf = os.path.join(params.working_dir, params.logging_file)

    logging.basicConfig(
        filename=pathf,
        encoding="utf-8",
        filemode="w",
        level=getattr(logging, logging_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    state.logger = logging.getLogger("my_logger")

    if not pathf == "":
        os.system("echo rm " + pathf + " >> clean.sh")


# Print parameters in screen and a dedicated file
def print_params(params):
    param_file = os.path.join(params.working_dir, "params_saved.json")

    # load the given parameters
    with open(param_file, "w") as json_file:
        json.dump(params.__dict__, json_file, indent=2)

    os.system("echo rm " + param_file + " >> clean.sh")
