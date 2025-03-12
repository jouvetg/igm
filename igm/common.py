#!/usr/bin/env python3

"""
# Copyright (C) 2021-2025 IGM authors 
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
import time

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
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm
import datetime

def setup_igm_modules(cfg, state) -> List[ModuleType]:
    return load_modules(cfg, state)


def initialize_modules(processes: List, cfg: Any, state: State) -> None:
    for module in processes:
        if cfg.core.logging:
            state.logger.info(f"Initializing module: {module.__name__.split('.')[-1]}")
        module.initialize(cfg, state)

def print_info(state):
 
    if state.it % 100 == 1:
        if hasattr(state, "pbar"):
            state.pbar.close()
        state.pbar = tqdm(desc=f"IGM", 
                          ascii=False, 
                          dynamic_ncols=True,
                          bar_format="{desc} {postfix}")   

    if hasattr(state, "pbar"):
        state.pbar.set_postfix({ 
            "🕒": datetime.datetime.now().strftime("%H:%M:%S"),
            "🔄": f"{state.it:06.0f}",
            "⏱ Time": f"{state.t.numpy():09.1f} yr",
            "⏳ Step": f"{state.dt_target:04.2f} yr",
            "❄️  Volume": f"{np.sum(state.thk) * (state.dx**2) / 10**9:108.2f} km³",
#            "💾 GPU Mem (MB)": tf.config.experimental.get_memory_info("GPU:0")['current'] / 1024**2
        })
        state.pbar.update(1)


def update_modules(processes: List, outputs: List, cfg: Any, state: State) -> None:
    if hasattr(state, "t"):
        if cfg.core.print_comp:
            state.tcomp = {module.__name__.split('.')[-1]: [] for module in processes+outputs}
        while state.t < cfg.processes.time.end:
            for module in processes:
                m=module.__name__.split('.')[-1]
                if cfg.core.print_comp:
                    state.tcomp[m].append(time.time())
                module.update(cfg, state)
                if cfg.core.print_comp:
                    state.tcomp[m][-1] -= time.time() ; state.tcomp[m][-1] *= -1
            run_outputs(outputs, cfg, state)
            if cfg.core.print_info:
                print_info(state)

def finalize_modules(processes: List, cfg: Any, state: State) -> None:
    for module in processes:
        module.finalize(cfg, state)


def run_outputs(output_modules: List, cfg: Any, state: State) -> None:
    for module in output_modules:
        m=module.__name__.split('.')[-1]
        if cfg.core.print_comp:
            state.tcomp[m].append(time.time())
        module.run(cfg, state)
        if cfg.core.print_comp:
            state.tcomp[m][-1] -= time.time() ; state.tcomp[m][-1] *= -1


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

def get_module_name(module):
        return module.__name__.split('.')[-1]

def get_orders():
    import yaml
    config_path = [path["path"] for path in HydraConfig.get().runtime.config_sources if path["schema"] == "file"][0]
    
    with open(f'{config_path}/experiment/{HydraConfig.get().runtime.choices.experiment}.yaml', 'r') as file:
        original_experiment_config = yaml.safe_load(file)
    
    defaults = original_experiment_config['defaults']
    input_order = modules_order = output_order = []
    for default in defaults:

        key = list(default.keys())[0] # ? Cleaner / more robust way to do this?
        if key == 'override /inputs':
            input_order = default[key]
        elif key == 'override /processes':
            modules_order = default[key]
        elif key == 'override /outputs':
            output_order = default[key]

    return input_order, modules_order, output_order
        

def load_modules(
    cfg, state
) -> Tuple[List[ModuleType], List[ModuleType], List[ModuleType]]:
    """Returns a list of actionable modules to then apply the update, initialize, finalize functions on for IGM."""
    imported_input_modules = []
    imported_modules = []
    imported_output_modules = []

    root_foldername = (
        f"{HydraConfig.get().runtime.cwd}/{cfg.core.structure.root_foldername}"
    )

    if "inputs" in cfg:
        user_input_modules_folder = f"{root_foldername}/{cfg.core.structure.code_foldername}/{cfg.core.structure.input_modules_foldername}"
        load_user_modules(
            cfg=cfg,
            state=state,
            modules_list=cfg.inputs,
            imported_modules_list=imported_input_modules,
            module_folder=user_input_modules_folder,
        )
        load_modules_igm(
            cfg=cfg,
            state=state,
            modules_list=cfg.inputs,
            imported_modules_list=imported_input_modules,
            module_type="inputs",
        )

    if "processes" in cfg:
        user_process_modules_folder = f"{root_foldername}/{cfg.core.structure.code_foldername}/{cfg.core.structure.process_modules_foldername}"
        load_user_modules(
            cfg=cfg,
            state=state,
            modules_list=cfg.processes,
            imported_modules_list=imported_modules,
            module_folder=user_process_modules_folder,
        )
        load_modules_igm(
            cfg=cfg,
            state=state,
            modules_list=cfg.processes,
            imported_modules_list=imported_modules,
            module_type="processes",
        )

    if "outputs" in cfg:
        user_output_modules_folder = f"{root_foldername}/{cfg.core.structure.code_foldername}/{cfg.core.structure.output_modules_foldername}"
        load_user_modules(
            cfg=cfg,
            state=state,
            modules_list=cfg.outputs,
            imported_modules_list=imported_output_modules,
            module_folder=user_output_modules_folder,
        )
        load_modules_igm(
            cfg=cfg,
            state=state,
            modules_list=cfg.outputs,
            imported_modules_list=imported_output_modules,
            module_type="outputs",
        )
    
    # Reorder modules
    input_order, module_order, output_order = get_orders()
    
    input_order_dict = {name: index for index, name in enumerate(input_order)}
    imported_input_modules = sorted(imported_input_modules, key=lambda module: input_order_dict[get_module_name(module)])
    
    modules_order_dict = {name: index for index, name in enumerate(module_order)}
    imported_modules = sorted(imported_modules, key=lambda module: modules_order_dict[get_module_name(module)])
    
    output_order_dict = {name: index for index, name in enumerate(output_order)}
    imported_output_modules = sorted(imported_output_modules, key=lambda module: output_order_dict[get_module_name(module)])

    if cfg.core.print_imported_modules:
        print(f"{'':-^100}")
        print(f"{'INPUTS Modules':-^100}")
        for i, input_module in enumerate(imported_input_modules):
            print(f" {i}: {input_module}")
        print(f"{'PROCESSES Modules':-^100}")
        for i, module in enumerate(imported_modules):
            print(f" {i}: {module}")
        print(f"{'OUTPUTS Modules':-^100}")
        for i, output_module in enumerate(imported_output_modules):
            print(f" {i}: {output_module}")
        print(f"{'':-^100}")

    return imported_input_modules, imported_modules, imported_output_modules


def load_user_modules(
    cfg, state, modules_list, imported_modules_list, module_folder
) -> List[ModuleType]:

    from importlib.machinery import SourceFileLoader

    # print("Testing for custom modules first - then will load default IGM modules...")
    for module_name in modules_list:
        # print("module name", module_name)
        # Local Directory
        try: 
            module = SourceFileLoader(
                f"{module_name}", f".{module_name}.py"
            ).load_module()
        except FileNotFoundError:
            # print(f'{module_name} [not found] in local working directory: {HydraConfig.get().runtime.cwd}. Trying custom modules directory...')

            # User Modules Folder
            try: 
                module = SourceFileLoader(
                    f"{module_name}", f"{module_folder}/{module_name}.py"
                ).load_module()
            except FileNotFoundError:
                # User Modules Folder Folder
                try:
                    module = SourceFileLoader(
                        f"{module_name}", f"{module_folder}/{module_name}/{module_name}.py"
                    ).load_module()
                except FileNotFoundError:
                    pass
                else:
                    # validate_module(module)
                    imported_modules_list.append(module)
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


def load_modules_igm(
    cfg, state, modules_list, imported_modules_list, module_type
) -> List[ModuleType]:

    from importlib.machinery import SourceFileLoader

    imported_modules_names = [module.__name__ for module in imported_modules_list]
    for module_name in modules_list:
        if module_name in imported_modules_names:
            continue

        module_path = f"igm.{module_type}.{module_name}"
        module = importlib.import_module(module_path)
        if module_type == "processes":
            validate_module(module)
        imported_modules_list.append(module)

def validate_module(module) -> None:
    """Validates that a module has the required functions to be used in IGM."""
    required_functions = ["initialize", "finalize", "update"]
    for function in required_functions:
        if not hasattr(module, function):
            raise AttributeError(
                f"Module {module} is missing the required function ({function}). If it is a custom python package, make sure to include the 3 required functions: ['initialize', 'finalize', 'update'].",
                f"Please see https://github.com/jouvetg/igm/wiki/5.-Custom-modules-(coding) for more information on how to construct custom modules.",
            )

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
    if not os.path.exists(folder_path):  # directory exists?
        os.makedirs(folder_path)

        # Download the file with wget
        logging.info("Downloading the ZIP file with wget...")
        subprocess.run(["wget", "-O", "downloaded_file.zip", url])

        # Unzipping the file
        logging.info("Unzipping the file...")
        with zipfile.ZipFile("downloaded_file.zip", "r") as zip_ref:
            zip_ref.extractall(folder_path)

        # Clean up (delete) the zip file after extraction
        os.remove("downloaded_file.zip")
        logging.info(f"File successfully downloaded and extracted to '{folder_path}'")

    else:
        logging.info(f"The data already exists at '{folder_path}'")


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

    modules = list(state.tcomp.keys())

    print("Computational statistics report:")
    with open("computational-statistics.txt", "w") as f:
        for m in modules:
            CELA = ( m, np.mean(state.tcomp[m]), np.sum(state.tcomp[m]) )
            print("     %14s  |  mean time per it : %8.4f  |  total : %8.4f" % CELA, file=f)
            print("     %14s  |  mean time per it : %8.4f  |  total : %8.4f" % CELA)

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

    modules = list(state.tcomp.keys())

    for m in modules:
        total.append(np.sum(state.tcomp[m][1:]))
        name.append(m)

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
