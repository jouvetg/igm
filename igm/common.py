#!/usr/bin/env python3

"""
Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
Published under the GNU GPL (Version 3), check at the LICENSE file
"""

import os, json, sys
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
import matplotlib.pyplot as plt
import numpy as np
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


def initialize_modules(modules: List, cfg: Any, state: State) -> None:
    for module in modules:
        print(f"Initializing module: {module.__name__.split('.')[-1]}")
        # print(dir(module))
        # print(module)
        # print(dir(state))
        module.initialize(cfg, state)
        # print(dir(state))
        # exit()


def run_model(modules: List, output_modules: List, cfg: Any, state: State) -> None:
    if hasattr(state, "t"):
        while state.t < cfg.modules.time.end:
            for module in modules:
                module.update(cfg, state)
            run_outputs(output_modules, cfg, state)


# def run_finalizers(modules: List, cfg: Any, state: State) -> None:
#     for module in modules:
#         module.finalize(cfg, state)

def run_outputs(output_modules: List, cfg: Any, state: State) -> None:
    for module in output_modules:
        module.run(cfg, state)


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
    imported_input_modules = []
    imported_modules = []
    imported_output_modules = []
    
    # print(cfg.output)
    # exit()
    # imported_modules = load_modules_from_directory(cfg, state, modules_list=cfg.modules)
    root_foldername = f"{HydraConfig.get().runtime.cwd}/{cfg.core.structure.root_foldername}"
    
    user_input_modules_folder = f"{root_foldername}/{cfg.core.structure.code_foldername}/{cfg.core.structure.input_modules_foldername}"
    
    load_user_modules(cfg, state, modules_list=cfg.input, imported_modules_list=imported_input_modules, module_folder=user_input_modules_folder)
    load_modules_igm(cfg, state, modules_list=cfg.input, imported_modules_list=imported_input_modules, module_type='input')
    
    
    user_process_modules_folder = f"{root_foldername}/{cfg.core.structure.code_foldername}/{cfg.core.structure.process_modules_foldername}"
    load_user_modules(cfg, state, modules_list=cfg.modules, imported_modules_list=imported_modules, module_folder=user_process_modules_folder)
    load_modules_igm(cfg, state, modules_list=cfg.modules, imported_modules_list=imported_modules, module_type='modules')
    
    # print(user_input_modules_folder)
    # exit()
    user_output_modules_folder = f"{root_foldername}/{cfg.core.structure.code_foldername}/{cfg.core.structure.output_modules_foldername}"
    load_user_modules(cfg, state, modules_list=cfg.output, imported_modules_list=imported_output_modules, module_folder=user_output_modules_folder)
    load_modules_igm(cfg, state, modules_list=cfg.output, imported_modules_list=imported_output_modules, module_type='output')
    
    print(f"{'':-^100}")
    print(f"{'INPUT Modules':-^100}")
    for i, input_module in enumerate(imported_input_modules):
        print(f' {i}: {input_module}')
    print(f"{'PHYSICAL Modules':-^100}")
    for i, module in enumerate(imported_modules):
        print(f' {i}: {module}')
    print(f"{'OUTPUT Modules':-^100}")
    for i, output_module in enumerate(imported_output_modules):
        print(f' {i}: {output_module}')
    print(f"{'':-^100}")
    
    return imported_input_modules, imported_modules, imported_output_modules

from hydra.core.hydra_config import HydraConfig
def load_user_modules(cfg, state, modules_list, imported_modules_list, module_folder) -> List[ModuleType]:
    
    from importlib.machinery import SourceFileLoader

    # print("Testing for custom modules first - then will load default IGM modules...")
    for module_name in modules_list:
        # Local Directory
        try:
            module = SourceFileLoader(f"{module_name}",f".{module_name}.py").load_module()
        except FileNotFoundError:
            # print(f'{module_name} [not found] in local working directory: {HydraConfig.get().runtime.cwd}. Trying custom modules directory...')
            
            # Custom Modules Folder
            try:
                module = SourceFileLoader(f"{module_name}",f"{module_folder}/{module_name}.py").load_module()
            except FileNotFoundError:
                pass
                # print(f'{module_name} [not found] in local directory or custom modules directory')
            else:
                # print(f'{module_name} [found] in custom modules directory: {HydraConfig.get().runtime.cwd}/{cfg.core.custom_modules_folder}')
                # validate_module(module)
                imported_modules_list.append(module)            
        else:
            # print(f'{module_name} [found] in local working directory: {HydraConfig.get().runtime.cwd}')
            # validate_module(module)
            imported_modules_list.append(module)
            
    return imported_modules_list

# def load_modules_custom(cfg, state, modules_list, imported_modules_list) -> List[ModuleType]:
    
#     from importlib.machinery import SourceFileLoader

#     # print("Testing for custom modules first - then will load default IGM modules...")
#     for module_name in modules_list:
#         # Local Directory
#         try:
#             module = SourceFileLoader(f"{module_name}",f".{module_name}.py").load_module()
#         except FileNotFoundError:
#             # print(f'{module_name} [not found] in local working directory: {HydraConfig.get().runtime.cwd}. Trying custom modules directory...')
            
#             # Custom Modules Folder
#             try:
#                 module = SourceFileLoader(f"{module_name}",f"{HydraConfig.get().runtime.cwd}/{cfg.core.custom_modules_folder}/{module_name}.py").load_module()
#             except FileNotFoundError:
#                 pass
#                 # print(f'{module_name} [not found] in local directory or custom modules directory')
#             else:
#                 # print(f'{module_name} [found] in custom modules directory: {HydraConfig.get().runtime.cwd}/{cfg.core.custom_modules_folder}')
#                 # validate_module(module)
#                 imported_modules_list.append(module)            
#         else:
#             # print(f'{module_name} [found] in local working directory: {HydraConfig.get().runtime.cwd}')
#             # validate_module(module)
#             imported_modules_list.append(module)
            
#     return imported_modules_list

# def load_modules_custom_output(cfg, state, modules_list, imported_modules_list) -> List[ModuleType]:
    
#     from importlib.machinery import SourceFileLoader

#     # print("Testing for custom modules first - then will load default IGM modules...")
#     for module_name in modules_list:
#         # Local Directory
#         try:
#             module = SourceFileLoader(f"{module_name}",f".{module_name}.py").load_module()
#         except FileNotFoundError:
#             # print(f'{module_name} [not found] in local working directory: {HydraConfig.get().runtime.cwd}. Trying custom modules directory...')
            
#             # Custom Modules Folder
#             try:
#                 module = SourceFileLoader(f"{module_name}",f"{HydraConfig.get().runtime.cwd}/{cfg.core.custom_output_modules_folder}/{module_name}.py").load_module()
#             except FileNotFoundError:
#                 pass
#                 # print(f'{module_name} [not found] in local directory or custom modules directory')
#             else:
#                 # print(f'{module_name} [found] in custom modules directory: {HydraConfig.get().runtime.cwd}/{cfg.core.custom_modules_folder}')
#                 # validate_module(module)
#                 imported_modules_list.append(module)            
#         else:
#             # print(f'{module_name} [found] in local working directory: {HydraConfig.get().runtime.cwd}')
#             # validate_module(module)
#             imported_modules_list.append(module)
            
#     return imported_modules_list
            
def load_modules_igm(cfg, state, modules_list, imported_modules_list, module_type) -> List[ModuleType]:
    
    from importlib.machinery import SourceFileLoader

    # print("Testing for custom modules first - then will load default IGM modules...")
    imported_modules_names = [module.__name__ for module in imported_modules_list]
    print(imported_modules_names)
    # exit()
    for module_name in modules_list:
        if module_name in imported_modules_names:
            continue
        
        module_path = f"igm.{module_type}.{module_name}"
        module = importlib.import_module(module_path)
        if module_type == 'modules':
            validate_module(module)
        imported_modules_list.append(module)

# def load_input_modules_igm(cfg, state, modules_list, imported_modules_list) -> List[ModuleType]:
    
#     from importlib.machinery import SourceFileLoader

#     # print("Testing for custom modules first - then will load default IGM modules...")
#     imported_modules_names = [module.__name__ for module in imported_modules_list]
#     for module_name in modules_list:
#         if module_name in imported_modules_names:
#             continue
        
#         module_path = f"igm.input.{module_name}"
#         module = importlib.import_module(module_path)
#         # validate_module(module)
#         imported_modules_list.append(module)

# def load_output_modules_igm(cfg, state, modules_list, imported_modules_list) -> List[ModuleType]:
    
#     from importlib.machinery import SourceFileLoader

#     # print("Testing for custom modules first - then will load default IGM modules...")
#     imported_modules_names = [module.__name__ for module in imported_modules_list]
#     for module_name in modules_list:
#         if module_name in imported_modules_names:
#             continue
        
#         module_path = f"igm.output.{module_name}"
#         module = importlib.import_module(module_path)
#         # validate_module(module)
#         imported_modules_list.append(module)
        
            
# def load_modules_custom_directory(module_name: str, custom_modules_folder: str):
#     print(f"{custom_modules_folder}.{module_name}")
#     module = importlib.import_module(f"{custom_modules_folder}.{module_name}")
    
def validate_module(module) -> None:
    """Validates that a module has the required functions to be used in IGM."""
    required_functions = ["initialize", "finalize", "update"]
    for function in required_functions:
        if not hasattr(module, function):
            raise AttributeError(
                f"Module {module} is missing the required function ({function}). If it is a custom python package, make sure to include the 3 required functions: ['initialize', 'finalize', 'update'].",
                f"Please see https://github.com/jouvetg/igm/wiki/5.-Custom-modules-(coding) for more information on how to construct custom modules.",
            )

# import warnings
# def import_custom_module(module_name: str, custom_modules_folder: str):
#     module = None
#     try:
#         module = importlib.import_module(module_name)
#     except ModuleNotFoundError:
#         print('here it is')
#         try:
#             module = importlib.import_module(f"{custom_modules_folder}.{module_name}")
#         except ModuleNotFoundError:
#             # raise ValueError(
#             # warnings.warn(
#                 # f"{module_name}. Make sure it is either in the 1) {Path(igm.__file__).parent}/modules/ directory or 2) in your current working directory."
#             # )
#             #     f"Can not find module {module_name}. Make sure it is either in the 1) {Path(igm.__file__).parent}/modules/ directory or 2) in your current working directory."
#             # )
#             pass

#     return module


# def load_modules_from_directory(  # CAREFUL -> OVERRIDING FUNCTIONS ARE NOT ACTUALLY HAPPENING! FIX!!!
#     cfg, state, modules_list: List[str]
# ) -> List[ModuleType]:
#     imported_modules = []
#     for module_name in modules_list:
#         module_path = f"igm.modules.{module_name}"

#         # print(module_path)
#         # ! CLEAN UP THIS CODE
#         # Look into best practices for logging and error handling (as this is a tricky sequence...)
#         try:
#             # print(module_path)
#             module = importlib.import_module(module_path)
#             # print(module)

#             # state.logger.info(
#             #     f"Module {module_name} exist in source code. Checking to see if an override is attempted."
#             # )
#             if module_name not in ['time']: # temporary solution here to get around the builtin module issue...
#                 custom_module = import_custom_module(
#                     module_name, custom_modules_folder=cfg.core.custom_modules_folder
#                 )
#                 print(module)
#                 print(custom_module)
#             else:
#                 custom_module = module
#             if custom_module is None:
#                 # state.logger.info(
#                 #     f"Using source code version for module {module_name}."
#                 # )
#                 pass
#             else:
#                 # state.logger.info(
#                 #     f"Custom module {module_name} has overridden the original module."
#                 # )
#                 # print('c', custom_module)
#                 module = custom_module
#                 # print(module)

#         except ModuleNotFoundError:
#             # state.logger.info(f"Module {module_name} does not exist in source code.")
#             custom_module_attempted = True

#             if custom_module_attempted:
#                 # state.logger.info(
#                     # f"Custom module {module_name} attempted. Attempting to import it."
#                 # )
#                 module = import_custom_module(
#                     module_name, custom_modules_folder=cfg.core.custom_modules_folder
#                 )
#                 # if module is not None:
#                     # state.logger.info(f"Custom module {module_name} successfully imported.")

#         if module is not None:
#             # state.logger.info(f"Validating {module_name} module.")
#             validate_module(module)
#             imported_modules.append(module)
#             print(module)

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


# def save_params(cfg, extension="yaml") -> None:
#     param_file = f"{cfg.saved_params_filename}.{extension}"
#     yaml_params = OmegaConf.to_yaml(cfg)
#     # load the given parameters
#     with open(param_file, "w") as file:
#         yaml.dump(yaml_params, file)


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

def print_comp(state):
    ################################################################

    size_of_tensor = {}

    for m in state.__dict__.keys():
        try:
            size_gb = sys.getsizeof(getattr(state, m).numpy())
            if size_gb > 1024**1:
                size_of_tensor[m] = size_gb / (1024**3)
        except:
            pass

    # sort from highest to lowest
    size_of_tensor = dict(
        sorted(size_of_tensor.items(), key=lambda item: item[1], reverse=True)
    )

    print("Memory statistics report:")
    with open("memory-statistics.txt", "w") as f:
        for key, value in size_of_tensor.items():
            print("     %24s  |  size : %8.4f Gb " % (key, value), file=f)
            print("     %24s  |  size : %8.4f Gb  " % (key, value))

    _plot_memory_pie(state)

    ################################################################

    modules = [A for A in state.__dict__.keys() if "tcomp_" in A]

    state.tcomp_all = [np.sum([np.sum(getattr(state, m)) for m in modules])]

    print("Computational statistics report:")
    with open("computational-statistics.txt", "w") as f:
        for m in modules:
            CELA = (
                m[6:],
                np.mean(getattr(state, m)),
                np.sum(getattr(state, m)),
                len(getattr(state, m)),
            )
            print(
                "     %24s  |  mean time per it : %8.4f  |  total : %8.4f  |  number it : %8.0f"
                % CELA,
                file=f,
            )
            print(
                "     %24s  |  mean time per it : %8.4f  |  total : %8.4f  |  number it  : %8.0f"
                % CELA
            )

    _plot_computational_pie(state)


def _plot_computational_pie(state):
    """
    Plot to the computational time of each model components in a pie
    """

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return "{:.0f}".format(val)

        return my_autopct

    total = []
    name = []

    modules = [A for A in state.__dict__.keys() if "tcomp_" in A]
    modules.remove("tcomp_all")

    for m in modules:
        total.append(np.sum(getattr(state, m)[1:]))
        name.append(m[6:])

    sumallindiv = np.sum(total)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"), dpi=200)
    wedges, texts, autotexts = ax.pie(
        total, autopct=make_autopct(total), textprops=dict(color="w")
    )
    ax.legend(
        wedges,
        name,
        title="Model components",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
    )
    plt.setp(autotexts, size=8, weight="bold")
    #    ax.set_title("Matplotlib bakery: A pie")
    plt.tight_layout()
    plt.savefig("computational-pie.png", pad_inches=0)
    plt.close("all")

def _plot_memory_pie(state):
    """
    Plot to the memory size of each model components in a pie
    """

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return "{:.0f}".format(val)

        return my_autopct

    size_of_tensor = {}

    for m in state.__dict__.keys():
        try:
            size_gb = sys.getsizeof(getattr(state, m).numpy())
            if size_gb > 1024**1:
                size_of_tensor[m] = size_gb / (1024**3)
        except:
            pass

    size_of_tensor = dict(
        sorted(size_of_tensor.items(), key=lambda item: item[1], reverse=True)
    )

    total = list(size_of_tensor.values())[:10]
    name = list(size_of_tensor.keys())[:10]

    sumallindiv = np.sum(total)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"), dpi=200)
    wedges, texts, autotexts = ax.pie(
        total, autopct=make_autopct(total), textprops=dict(color="w")
    )
    ax.legend(
        wedges,
        name,
        title="Model components",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
    )
    plt.setp(autotexts, size=8, weight="bold")
    #    ax.set_title("Matplotlib bakery: A pie")
    plt.tight_layout()
    plt.savefig("memory-pie.png", pad_inches=0)
    plt.close("all")

