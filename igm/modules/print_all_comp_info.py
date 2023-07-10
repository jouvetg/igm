#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
This serves to print computational info report

==============================================================================

Input  : .....
Output : .....
"""

import numpy as np
import os, sys, shutil
import time
import tensorflow as tf
import matplotlib.pyplot as plt


def params_print_all_comp_info(parser):
    pass

def init_print_all_comp_info(params, self):
    pass

def update_print_all_comp_info(params, self):
    pass

def final_print_all_comp_info(params, self):
 
    modules = [A for A in self.__dict__.keys() if 'tcomp_' in A]

    self.tcomp_all = [ np.sum([np.sum(getattr(self,m)) for m in modules]) ]

    print("Computational statistics report:")
    with open(
        os.path.join(params.working_dir, "computational-statistics.txt"), "w"
    ) as f:
        for m in modules:
            CELA = (
                m[6:],
                np.mean(getattr(self,m)),
                np.sum(getattr(self,m)),
                len(getattr(self,m)),
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

    os.system(
        "echo rm "
        + os.path.join(params.working_dir, "computational-statistics.txt")
        + " >> clean.sh"
    )

    _plot_computational_pie(params, self)

def _plot_computational_pie(params, self):
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

    modules = [A for A in self.__dict__.keys() if 'tcomp_' in A]
    modules.remove('tcomp_all')

    for m in modules:
        total.append(np.sum(getattr(self,m)[1:]))
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
    plt.savefig(os.path.join(params.working_dir, "computational-pie.png"), pad_inches=0)
    plt.close("all")

    os.system(
        "echo rm "
        + os.path.join(params.working_dir, "computational-pie.png")
        + " >> clean.sh"
    )