#!/usr/bin/env python3

"""
Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
Published under the GNU GPL (Version 3), check at the LICENSE file
"""

import os, glob

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

# read a file and return a list of words
def _read_file_to_list(file_path):
    word_list = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                 # Remove leading/trailing whitespaces and newline characters
                word = line.strip() 
                word_list.append(word)
    return word_list

# load modules from files
def load_modules():

    L = glob.glob("modules_*.dat")

    if len(L)>0:
        modules_preproc  = _read_file_to_list('modules_preproc.dat')
        modules_process  = _read_file_to_list('modules_process.dat')
        modules_postproc = _read_file_to_list('modules_postproc.dat')
    else:
        modules_preproc  = ["prepare_data"]
        modules_process  = ["mysmb","flow_dt_thk"]
        modules_preproc  = ["write_ncdf_ex","write_plot2d","print_info","print_all_comp_info"]
    
    return modules_preproc, modules_process, modules_postproc