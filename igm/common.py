#!/usr/bin/env python3

"""
Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
Published under the GNU GPL (Version 3), check at the LICENSE file
"""

import os, json
from json import JSONDecodeError
import importlib
import argparse
from argparse import ArgumentParser, Namespace
from pathlib import Path
from functools import partial
from typing import List, Any, Dict, Tuple
from types import ModuleType
import logging
import warnings

import tensorflow as tf

import igm

IGM_DESCRIPTION = r"""
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │             Welcome to IGM, a modular, open-source, fast, and user-friendly glacier evolution model!             │
  │                                                                                                                  │
  │                                                                                                                  │
  │                         __/\\\\\\\\\\\_____/\\\\\\\\\\\\__/\\\\____________/\\\\_                                │
  │                          _\/////\\\///____/\\\//////////__\/\\\\\\________/\\\\\\_                               │
  │                           _____\/\\\______/\\\_____________\/\\\//\\\____/\\\//\\\_                              │
  │                            _____\/\\\_____\/\\\____/\\\\\\\_\/\\\\///\\\/\\\/_\/\\\_                             │
  │                             _____\/\\\_____\/\\\___\/////\\\_\/\\\__\///\\\/___\/\\\_                            │
  │                              _____\/\\\_____\/\\\_______\/\\\_\/\\\____\///_____\/\\\_                           │
  │                               _____\/\\\_____\/\\\_______\/\\\_\/\\\_____________\/\\\_                          │
  │                                __/\\\\\\\\\\\_\//\\\\\\\\\\\\/__\/\\\_____________\/\\\_                         │
  │                                 _\///////////___\////////////____\///______________\///__                        │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""


class State:
    pass


from omegaconf import DictConfig, OmegaConf


def setup_igm_modules(cfg, state) -> List[ModuleType]:
    return load_modules(cfg, state)


def run_intializers(modules: List, cfg: Any, state: State) -> None:
    for module in modules:
        print(f"Initializing module: {module.__name__.split('.')[-1]}")
        # print(dir(module))
        # print(module)
        # print(dir(state))
        module.initialize(cfg, state)
        # print(dir(state))
        # exit()


def run_processes(modules: List, cfg: Any, state: State) -> None:
    if hasattr(state, "t"):
        while state.t < cfg.modules.time_igm.time_end:
            for module in modules:
                module.update(cfg, state)


def run_finalizers(modules: List, cfg: Any, state: State) -> None:
    for module in modules:
        module.finalize(cfg, state)


def add_logger(cfg, state) -> None:

    # ! Ignore logging file for now...
    # if cfg.logging_file == "":
    #     pathf = ""
    # else:
    #     pathf = cfg.logging_file

    logging.basicConfig(
        # filename=pathf,
        encoding="utf-8",
        filemode="w",
        level=cfg.core.logging_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.root.setLevel(cfg.core.logging_level)

    state.logger = logging.getLogger("igm")


def load_modules(cfg, state) -> List[ModuleType]:
    """Returns a list of actionable modules to then apply the update, initialize, finalize functions on for IGM."""
    imported_modules = load_modules_from_directory(cfg, state, modules_list=cfg.modules)

    return imported_modules


def validate_module(module) -> None:
    """Validates that a module has the required functions to be used in IGM."""
    required_functions = ["initialize", "finalize", "update"]
    for function in required_functions:
        if not hasattr(module, function):
            raise AttributeError(
                f"Module {module} is missing the required function ({function}). If it is a custom python package, make sure to include the 3 required functions: ['initialize', 'finalize', 'update'].",
                f"Please see https://github.com/jouvetg/igm/wiki/5.-Custom-modules-(coding) for more information on how to construct custom modules.",
            )

import warnings
def import_custom_module(module_name: str, custom_modules_folder: str):
    module = None
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        try:
            module = importlib.import_module(f"{custom_modules_folder}.{module_name}")
        except ModuleNotFoundError:
            # raise ValueError(
            # warnings.warn(
                # f"{module_name}. Make sure it is either in the 1) {Path(igm.__file__).parent}/modules/ directory or 2) in your current working directory."
            # )
            #     f"Can not find module {module_name}. Make sure it is either in the 1) {Path(igm.__file__).parent}/modules/ directory or 2) in your current working directory."
            # )
            pass

    return module


def load_modules_from_directory(  # CAREFUL -> OVERRIDING FUNCTIONS ARE NOT ACTUALLY HAPPENING! FIX!!!
    cfg, state, modules_list: List[str]
) -> List[ModuleType]:
    imported_modules = []
    for module_name in modules_list:
        module_path = f"igm.modules.{module_name}"

        # ! CLEAN UP THIS CODE
        # Look into best practices for logging and error handling (as this is a tricky sequence...)
        try:
            module = importlib.import_module(module_path)

            # state.logger.info(
            #     f"Module {module_name} exist in source code. Checking to see if an override is attempted."
            # )
            custom_module = import_custom_module(
                module_name, custom_modules_folder=cfg.core.custom_modules_folder
            )
            if custom_module is None:
                # state.logger.info(
                #     f"Using source code version for module {module_name}."
                # )
                pass
            else:
                # state.logger.info(
                #     f"Custom module {module_name} has overridden the original module."
                # )
                module = custom_module

        except ModuleNotFoundError:
            # state.logger.info(f"Module {module_name} does not exist in source code.")
            custom_module_attempted = True

            if custom_module_attempted:
                # state.logger.info(
                    # f"Custom module {module_name} attempted. Attempting to import it."
                # )
                module = import_custom_module(
                    module_name, custom_modules_folder=cfg.core.custom_modules_folder
                )
                # if module is not None:
                    # state.logger.info(f"Custom module {module_name} successfully imported.")

        if module is not None:
            # state.logger.info(f"Validating {module_name} module.")
            validate_module(module)
            imported_modules.append(module)

    return imported_modules


# def load_modules_from_directory( # CAREFUL -> OVERRIDING FUNCTIONS ARE NOT ACTUALLY HAPPENING! FIX!!!
#     modules_list: List[str]
# ) -> List[ModuleType]:
#     imported_modules = []
#     for module_name in modules_list:
#         module_path = f"igm.modules.{module_name}"

#         try:
#             module = importlib.import_module(module_path)
#             print(f"Module {module_name} exist in source code.")
#             try:
#                 # test if one is trying to overrite existing module...

#         except ModuleNotFoundError:
#             logging.info(
#                 f"Error importing module: {module_path}, checking for custom package in current working directory."
#             )
#             try:
#                 logging.info(
#                     f"Trying to import custom module from current working directory (folder or .py): {module_name}"
#                 )
#                 try:
#                     module = importlib.import_module(module_name)
#                 except ModuleNotFoundError:
#                     module = importlib.import_module("modules_custom." + module_name)
#             except ModuleNotFoundError:
#                 raise ModuleNotFoundError(
#                     f"Can not find module {module_name}. Make sure it is either in the 1) {Path(igm.__file__).parent}/modules/ directory or 2) in your current working directory."
#                 )

#         validate_module(module)
#         imported_modules.append(module)

#     return imported_modules


def print_gpu_info() -> None:
    gpus = tf.config.experimental.list_physical_devices("GPU")
    print(f"{'CUDA Enviroment':-^150}")
    tf.sysconfig.get_build_info().pop("cuda_compute_capabilities", None)
    print(f"{json.dumps(tf.sysconfig.get_build_info(), indent=2, default=str)}")
    print(f"{'Available GPU Devices':-^150}")
    for gpu in gpus:
        gpu_info = {"gpu_id": gpu.name, "device_type": gpu.device_type}
        device_details = tf.config.experimental.get_device_details(gpu)
        gpu_info.update(device_details)

        print(f"{json.dumps(gpu_info, indent=2, default=str)}")
    print(f"{'':-^150}")


def save_params(cfg, extension="yaml") -> None:
    param_file = f"{cfg.saved_params_filename}.{extension}"
    yaml_params = OmegaConf.to_yaml(cfg)
    # load the given parameters
    with open(param_file, "w") as file:
        yaml.dump(yaml_params, file)


def download_unzip_and_store(url, folder_path) -> None:
    """
    Use wget to download a ZIP file and unzip its contents to a specified folder.

    Args:
    - url (str): The URL of the ZIP file to download.
    - folder_path (str): The path of the folder where the ZIP file's contents will be extracted.
    # - folder_name (str): The name of the folder where the ZIP file's contents will be extracted.
    """

    import subprocess
    import os
    import zipfile

    # Ensure the destination folder exists
    if not os.path.exists(folder_path): # directory exists?
        os.makedirs(folder_path)

        # Download the file with wget
        print("Downloading the ZIP file with wget...")
        subprocess.run(["wget", "-O", "downloaded_file.zip", url])

        # Unzipping the file
        print("Unzipping the file...")
        with zipfile.ZipFile("downloaded_file.zip", "r") as zip_ref:
            zip_ref.extractall(folder_path)

        # Clean up (delete) the zip file after extraction
        os.remove("downloaded_file.zip")
        print(f"File successfully downloaded and extracted to '{folder_path}'")

    else:
        print(f"The data already exists at '{folder_path}'")
