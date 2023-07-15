#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
This IGM module wraps up module iceflow, time_step and thk.

==============================================================================

Takes  : ice thickness
Return : update ice flow, ice thickness, and time step
"""

from igm.modules.iceflow import *
from igm.modules.time_step import *
from igm.modules.thk import *


def params_flow_dt_thk(parser):
    params_iceflow(parser)
    params_time_step(parser)
    params_thk(parser)


def init_flow_dt_thk(params, state):
    init_iceflow(params, state)
    init_time_step(params, state)
    init_thk(params, state)


def update_flow_dt_thk(params, state):
    update_iceflow(params, state)
    update_time_step(params, state)
    update_thk(params, state) 


def final_flow_dt_thk(params, state):
    final_iceflow(params, state)
    final_time_step(params, state)
    final_thk(params, state) 
