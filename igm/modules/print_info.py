#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
This IGM modules prints basic information of IGM live while IGM is running

==============================================================================

Output: informartion on the fly about ice volume, time steps, ect.
"""

import numpy as np
import os
import datetime
import matplotlib.pyplot as plt


def params_print_info(self):
    pass


def init_print_info(params, self):

    print(
        "IGM %s :         Iterations   |         Time (y)     |     Time Step (y)   |   Ice Volume (km^3) "
    )


def update_print_info(params, self):
    """
    This serves to print key info on the fly during computation
    """
    if self.saveresult:
        print(
            "IGM %s :      %6.0f    |      %8.0f        |     %7.2f        |     %10.2f "
            % (
                datetime.datetime.now().strftime("%H:%M:%S"),
                self.it,
                self.t,
                self.dt_target,
                np.sum(self.thk) * (self.dx**2) / 10**9,
            )
        )
        
        
def final_print_info(params, self):
    pass

