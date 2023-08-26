#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

from igm.modules.process.iceflow import *
from igm.modules.process.time_step import *
from igm.modules.process.thk import *


def params_flow_dt_thk(parser):
    params_iceflow(parser)
    params_time_step(parser)
    params_thk(parser)


def initialize_flow_dt_thk(params, state):
    initialize_iceflow(params, state)
    initialize_time_step(params, state)
    initialize_thk(params, state)


def update_flow_dt_thk(params, state):
    update_iceflow(params, state)
    update_time_step(params, state)
    update_thk(params, state) 


def finalize_flow_dt_thk(params, state):
    finalize_iceflow(params, state)
    finalize_time_step(params, state)
    finalize_thk(params, state) 
