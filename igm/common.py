#!/usr/bin/env python3

"""
Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
Published under the GNU GPL (Version 3), check at the LICENSE file
"""

import os, json
from json import JSONDecodeError
import importlib
import argparse
import igm
import tensorflow as tf
from pathlib import Path
from functools import partial
from typing import List, Any, Dict
import types
import logging
import warnings


class State:
    pass


# this create core parameters for any IGM run
def params_core():
    parser = argparse.ArgumentParser(
        description="IGM", conflict_handler="resolve"
    )  # automatically overrides repeated/older parameters! Vaid solution?

    parser.add_argument(
        "--working_dir",
        type=str,
        default="",
        help="Working directory (default empty string)",
    )
    parser.add_argument(
        "--param_file",
        type=str,
        default="params.json",
        help="Path for the JSON parameter file.",
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
        action="store_true",
        default=False,
        help="Activate the logging (default value is False)",
    )
    parser.add_argument(
        "--logging_level",
        default=30,
        type=int,
        help="Determine logging level used for logger",
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
        "--gpu_id",
        type=int,
        default=0,
        help="Id of the GPU to use (default is 0)",
    )
    return parser


def run_intializers(modules: List, params: Any, state: State) -> None:
    for module in modules:
        module.initialize(params, state)


def run_processes(modules: List, params: Any, state: State) -> None:
    if hasattr(state, "t"):
        while state.t < params.time_end:
            for module in modules:
                module.update(params, state)


def run_finalizers(modules: List, params: Any, state: State) -> None:
    for module in modules:
        module.finalize(params, state)


def add_logger(params, state, logging_level) -> None:
    if params.logging_file == "":
        pathf = ""
    else:
        pathf = os.path.join(params.working_dir, params.logging_file)

    logging.basicConfig(
        filename=pathf,
        encoding="utf-8",
        filemode="w",
        level=logging_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.root.setLevel(logging_level)

    state.logger = logging.getLogger("igm_logger")
    if not pathf == "":
        os.system("echo rm " + pathf + " >> clean.sh")


def remove_comments(json_str) -> str:
    lines = json_str.split("\n")
    cleaned_lines = [
        line for line in lines if not line.strip().startswith(("//", "#"))
    ]  # ! TODO: Add blocks comments...
    cleaned_text = "\n".join(cleaned_lines)
    return cleaned_text


def get_modules_list(params_path: str) -> Dict[str, List[str]]:
    try:
        with open(params_path) as f:
            # params_dict = json.load(f) #re-instate if you want to enforce no comments in the json file(remove all the lines that load the clean json in this case...)
            json_text = f.read()
            json_cleaned = remove_comments(json_text)
            params_dict = json.loads(json_cleaned)
            module_dict = {
                "modules_preproc": params_dict["modules_preproc"],
                "modules_process": params_dict["modules_process"],
                "modules_postproc": params_dict["modules_postproc"],
            }

            return module_dict
    except JSONDecodeError as e:
        raise JSONDecodeError(
            msg="For the following line, please check the 1) JSON file structure and/or 2) make sure there are no comments (//, #, etc.)",
            doc=e.doc,
            pos=e.pos,
        )


def load_json_file(param_file: str, remove_param_comments: bool = True) -> Dict[str, Any]:
    with open(param_file, "r") as json_file:
        json_text = json_file.read()

    if remove_param_comments:
        json_text = remove_comments(json_text)

    dic_params = json.loads(json_text)
    return dic_params


def load_user_defined_params(param_file: str, params_dict: Dict[str, Any]):
    try:
        json_defined_params = load_json_file(param_file=param_file)
    except JSONDecodeError as e:
        raise JSONDecodeError(
            msg=f"Error decoding JSON: Please make sure your file path is correct or your json is properly formatted.",
            doc=e.doc,
            pos=e.pos,
        )

    unrecognized_json_arguments = {
        k: v for k, v in json_defined_params.items() if k not in params_dict
    }
    params_dict.update(json_defined_params)

    for key in unrecognized_json_arguments.keys():
        warnings.warn(
            f"The following argument specified in the JSON file does not exist among the core arguments nor the modules you have chosen. Ignoring: {key}"
        )

    return params_dict


def load_modules(modules_dict: Dict) -> List[types.ModuleType]:
    """Returns a list of actionable modules to then apply the update, initialize, finalize functions on for IGM."""

    imported_preproc_modules = load_modules_from_directory(
        modules_list=modules_dict["modules_preproc"], module_folder="preproc"
    )
    imported_process_modules = load_modules_from_directory(
        modules_list=modules_dict["modules_process"], module_folder="process"
    )
    imported_postproc_modules = load_modules_from_directory(
        modules_list=modules_dict["modules_postproc"], module_folder="postproc"
    )
    # ? Should we have custom modules in a seperate folder?
    # imported_custom_modules = load_modules_from_directory(
    #     modules_list=modules_dict.modules_custom, module_folder=params.modules_custom_folder
    # )

    return (
        imported_preproc_modules
        + imported_process_modules
        + imported_postproc_modules  # + imported_custom_modules
    )


def validate_module(module) -> None:
    """Validates that a module has the required functions to be used in IGM."""
    required_functions = ["params", "initialize", "finalize", "update"]
    for function in required_functions:
        if not hasattr(module, function):
            raise AttributeError(
                f"Module {module} is missing the required function ({function}). If it is a custom python package, make sure to include it in the __init__.py file.",
                f"Please see https://github.com/jouvetg/igm/wiki/5.-Custom-modules-(coding) for more information on how to construct custom modules.",
            )


def load_modules_from_directory(
    modules_list: List[str], module_folder: str
) -> List[types.ModuleType]:
    imported_modules = []
    for module_name in modules_list:
        module_path = f"igm.modules.{module_folder}.{module_name}"
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError:
            logging.info(
                f"Error importing module: {module_path}, checking for custom package in current working directory."
            )
            try:
                logging.info(
                    f"Trying to import custom module from current working directory (folder or .py): {module_name}"
                )
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    f"Can not find module {module_name}. Make sure it is either in the 1) {Path(igm.__file__).parent}/modules/{module_folder} directory or 2) in your current working directory."
                )

        validate_module(module)
        imported_modules.append(module)

    return imported_modules


def has_dependencies(module: Any) -> bool:
    if hasattr(module, "dependencies"):
        return True
    return False


def load_dependent_modules(imported_modules: List) -> List[types.ModuleType]:
    imported_dependencies = set()
    for module in imported_modules:
        if has_dependencies(module):
            module_dependencies = module.dependencies
            directories_to_search = [
                "preproc",
                "process",
                "postproc",
            ]  # will also check current working directory if any of them fail
            load_modules_partial = partial(
                load_modules_from_directory, module_dependencies
            )
            for directory in directories_to_search:
                try:
                    dependent_module = load_modules_partial(module_folder=directory)
                    imported_dependencies.add(
                        dependent_module[0]
                    )  # [0] because it returns a singleton list
                    logging.info(
                        f"Found dependencies in directory {directory} or current working directory. Checking next module for dependencies."
                    )
                    break
                except ModuleNotFoundError:
                    logging.info(
                        f"Could not find dependencies in directory {directory} or current working directory. Checking next IGM directory."
                    )

    return imported_modules + list(imported_dependencies)


def gpu_information() -> None:
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


# Print parameters in screen and a dedicated file
def print_params(params):
    param_file = os.path.join(params.working_dir, "params_saved.json")

    # load the given parameters
    with open(param_file, "w") as json_file:
        json.dump(params.__dict__, json_file, indent=2)

    os.system("echo rm " + param_file + " >> clean.sh")
