#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import datetime, time
import tensorflow as tf

from igm.modules.utils import compute_divflux_slope_limiter, compute_divflux, compute_divflux_adaptive
# from igm.tests.Bueler2005 import initialize as initialize_Bueler

from dataclasses import dataclass, field
import xarray as xr
from typing import Any, Tuple, Union, Optional


@dataclass
class OutputVariable:
    """Every Variable you want to be part of the output, should be defined as an instance of this class"""

    data: Any
    name: str
    attrs: dict = field(default_factory=dict)
    dims: dict = field(
        default_factory=dict
    )  # Optional Field to override the default dims
    # coords: Optional[str] = None # Optional Field to override the default coords
    # dims: Optional[str] = None # Optional Field to override the default dims
    
def params(parser):
    parser.add_argument(
        "--thk_slope_type",
        type=str,
        default="superbee",
        help="Type of slope limiter for the ice thickness equation (godunov or superbee)",
    )
    parser.add_argument(
        "--thk_ratio_density",
        type=float,
        default=0.910,
        help="density of ice divided by density of water",
    )
    parser.add_argument(
        "--thk_default_sealevel",
        type=float,
        default=0.0,
        help="Default sea level if not provided by the user",
    )

def initialize(cfg, state):

    if not hasattr(state, "topg"):
        raise ValueError("The 'thk' module requires an initial topography ('state.topg') to be defined. Please define it through the preprocessing steps (not yet implemented)")
        
    # define the lower ice surface
    if hasattr(state, "sealevel"):
        # ! This is not clear which modules provides state.topg and allows us to use state.thk!!!
        state.lsurf = tf.maximum(state.topg,-cfg.modules.thk.ratio_density*state.thk + state.sealevel)
    else:
        state.lsurf = tf.maximum(state.topg,-cfg.modules.thk.ratio_density*state.thk + cfg.modules.thk.default_sealevel)

    # define the upper ice surface
    state.usurf = state.lsurf + state.thk

    state.tcomp_thk = []

def update(cfg, state):

    if state.it >= 0:
        if hasattr(state, "logger"):
            state.logger.info(
                "Ice thickness equation at time : " + str(state.t.numpy())
            )

        state.tcomp_thk.append(time.time())

        # compute the divergence of the flux
        # state.divflux, Qx, Qy = compute_divflux_adaptive(
        #     state.ubar, state.vbar, state.thk, state.dx, state.dx
        # )
        # compute the divergence of the flux
        state.divflux, Qx, Qy = compute_divflux_slope_limiter(
            state.ubar, state.vbar, state.thk, state.dx, state.dx, state.dt, slope_type=cfg.modules.thk.slope_type
        )
        
        
        # print(Qx.shape)
        # print(Qy.shape)
        # print(Qx[..., :-1].shape)
        # print(Qy[1:, ...].shape)
        
        Qx_trimmed = Qx[..., 1:]
        Qy_trimmed = Qy[:-1, ...]
        state.Qx_here = Qx_trimmed
        state.Qy_here = Qy_trimmed
        

        # if not smb model is given, set smb to zero
        if not hasattr(state, "smb"):
            state.smb = tf.zeros_like(state.thk)

        # Forward Euler with projection to keep ice thickness non-negative
        state.thk = tf.maximum(state.thk + state.dt * (state.smb - state.divflux), 0)

        # # Demo of how to create an OutputVariable (will also have some default ones from the input...)
        # thk = state.thk
        # attributes = {
        #     "long_name": "Ice Thickness",
        #     "units": "m",
        #     "comments": "This comes from Isosia",
        # }
        
        # output_variable = OutputVariable(data=thk, name="thk", attrs=attributes)
        # state.output_variables.add_variable(output_variable)
        # import numpy as np
        # state.data_arrays["thk"].assign_coords(time=("time", np.reshape(state.t.numpy(), (1,))))
        
        # define the lower ice surface
        if hasattr(state, "sealevel"):
            state.lsurf = tf.maximum(state.topg,-cfg.modules.thk.ratio_density*state.thk + state.sealevel)
        else:
            state.lsurf = tf.maximum(state.topg,-cfg.modules.thk.ratio_density*state.thk + cfg.modules.thk.default_sealevel)

        # define the upper ice surface
        state.usurf = state.lsurf + state.thk

        state.tcomp_thk[-1] -= time.time()
        state.tcomp_thk[-1] *= -1


def finalize(params, state):
    pass
