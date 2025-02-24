#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file 

"""

Mass balance forced by climate with accumulation and temperature-index melt model [Hock, 1999; Huss et al., 2009]. 
It is a TensorFlow re-implementation of the model used in simulations reported in
(Jouvet and al., Numerical simulation of Rhone's glacier from 1874 to 2100, JCP, 2009)
(Jouvet and al., Modelling the retreat of Grosser Aletschgletscher in a changing climate, JOG, 2011)
(Jouvet and Huss, Future retreat of great Aletsch glacier, JOG, 2019)
(Jouvet and al., Mapping the age of ice of Gauligletscher ..., TC, 2020)
Check at this latest paper for the model description corresponding to this implementation

==============================================================================

Input: ---
Output: state.smb
"""

# Import the most important libraries
import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from netCDF4 import Dataset
from scipy.interpolate import interp1d
import time

 ## add custumized smb function
# def params(parser):  

#     parser.add_argument(
#         "--update_freq",
#         type=float,
#         default=1,
#         help="Update the mass balance each X years (1)",
#     )
#     parser.add_argument(
#         "--weight_ablation", 
#         type=float, 
#         default=1.0, 
#         help="Weight for melt",
#     )
#     parser.add_argument(
#         "--weight_accumulation",
#         type=float,
#         default=1.0,
#         help="Weight for accumulation",
#     )
#     parser.add_argument(
#         "--thr_temp_snow",
#         type=float,
#         default=0.5,
#         help="Threshold temperature for solid precipitation (0.0)",
#     )
#     parser.add_argument(
#         "--thr_temp_rain",
#         type=float,
#         default=2.5,
#         help="Threshold temperature for liquid precipitation (2.0)",
#     )
#     parser.add_argument(
#         "--shift_hydro_year",
#         type=float,
#         default=0.75,
#         help="This serves to start Oct 1. the acc/melt computation (0.75)",
#     )

#     parser.add_argument(
#         "--weight_Aletschfirn",
#         type=float,
#         default=1.0, 
#     )
#     parser.add_argument(
#         "--weight_Jungfraufirn",
#         type=float,
#         default=1.0, 
#     )
#     parser.add_argument(
#         "--weight_Ewigschneefeld",
#         type=float,
#         default=1.0, 
#     )
    
#     parser.add_argument(
#         "--ice_density",
#         type=float,
#         default=910.0,
#         help="Density of ice for conversion of SMB into ice equivalent",
#     )
#     parser.add_argument(
#         "--wat_density",
#         type=float,
#         default=1000.0,
#         help="Density of water",
#     )


def initialize(cfg,state):
    """
        load smb data to run the Aletsch Glacier simulation 
    """

    filepath = state.original_cwd.joinpath(os.path.join("data", 'massbalance.nc'))
    nc = Dataset(filepath, "r")
    x = np.squeeze(nc.variables["x"]).astype("float32")
    y = np.squeeze(nc.variables["y"]).astype("float32")

    state.snow_redistribution = np.squeeze(nc.variables["snow_redistribution"]
    ).astype("float32")
    state.direct_radiation = np.squeeze(nc.variables["direct_radiation"]).astype(
        "float32"
    )
    nc.close()
    
    
    nc = Dataset(state.original_cwd.joinpath(os.path.join("data", 'bassin.nc')), "r" )
    x = np.squeeze(nc.variables["x"]).astype("float32")
    y = np.squeeze(nc.variables["y"]).astype("float32")

    state.Aletschfirn = np.squeeze( nc.variables["Aletschfirn"] ).astype("float32")
    state.Ewigschneefeld = np.squeeze( nc.variables["Ewigschneefeld"] ).astype("float32")
    state.Jungfraufirn = np.squeeze( nc.variables["Jungfraufirn"] ).astype("float32")
 
    nc.close()


    if not hasattr(state, "x"):
        state.x = tf.constant(x, dtype="float32")
        state.y = tf.constant(y, dtype="float32")
    else:
        assert x.shape == state.x.shape
        assert y.shape == state.y.shape

    # resample at daily resolution the direct radiation field, 
    da = np.arange(20, 351, 30)
    db = np.arange(1, 366)
    state.direct_radiation = interp1d(
        da, state.direct_radiation, kind="linear", axis=0, fill_value="extrapolate")(db)
            
    state.direct_radiation = tf.Variable(state.direct_radiation, dtype="float32")

    # read mass balance parameters
    state.mb_parameters = tf.Variable(
        np.loadtxt(
            state.original_cwd.joinpath(os.path.join("data", 'mbparameter.dat')),
            skiprows=2,
            dtype=np.float32,
        )
    )

    # This permits to give some weight to some of the accumaulation bassins
    if not ((cfg.modules.smb_accmelt.weight_Aletschfirn==1.0)&(cfg.modules.smb_accmelt.weight_Jungfraufirn==1.0)&(cfg.modules.smb_accmelt.weight_Ewigschneefeld==1.0)):
        state.snow_redistribution *= (1.0/3.0)* \
            ( cfg.modules.smb_accmelt.weight_Aletschfirn    * state.Aletschfirn     + (1-state.Aletschfirn) \
            + cfg.modules.smb_accmelt.weight_Jungfraufirn   * state.Jungfraufirn    + (1-state.Jungfraufirn) \
            + cfg.modules.smb_accmelt.weight_Ewigschneefeld * state.Ewigschneefeld  + (1-state.Ewigschneefeld) )

    # snow_redistribution must have shape like precipitation (365,ny,nx)
    state.snow_redistribution = tf.expand_dims(state.snow_redistribution, axis=0)
    state.snow_redistribution = tf.tile(state.snow_redistribution, (365, 1, 1))

    # define the year-corresponding indice in the mb_parameters file
    state.IMB = np.zeros((221), dtype="int32")
    for tt in range(1880, 2101):
        state.IMB[tt - 1880] = np.argwhere(
            (state.mb_parameters[:, 0] <= tt) & (tt < state.mb_parameters[:, 1])
        )[0]
    state.IMB = tf.Variable(state.IMB)

    if "time" not in cfg.modules:
        raise ValueError("The 'time' module is required to use the 'smb_accmelt' module.")
    
    
    # ! We need an initial smb value as particles comes after in the update order...
    state.smb = tf.ones_like(state.topg) * (cfg.modules.smb_accmelt.wat_density / cfg.modules.smb_accmelt.ice_density)
    
    state.tlast_smb = tf.Variable(cfg.modules.time.start)
    state.tcomp_smb = []

# Warning: The decorator permits to take full benefit from efficient TensorFlow operation (especially on GPU)
# Note that tf.function works best with TensorFlow ops; NumPy and Python calls are converted to constants.
# Therefore: you must make sure any variables are TensorFlow Tensor (and not Numpy)
# @tf.function()
def update(cfg,state):

    if ((state.t - state.tlast_smb) >= cfg.modules.smb_accmelt.update_freq):

        if hasattr(state, "logger"):
            state.logger.info("update smb at time : " + str(state.t.numpy()))

        state.tcomp_smb.append(time.time())

        # update melt paramters
        Fm = state.mb_parameters[state.IMB[int(state.t) - 1880], 2] * 10 ** (-3)
        ri = state.mb_parameters[state.IMB[int(state.t) - 1880], 3] * 10 ** (-5)
        rs = state.mb_parameters[state.IMB[int(state.t) - 1880], 4] * 10 ** (-5)

        # keep solid precipitation when temperature < thr_temp_snow
        # with linear transition to 0 between thr_temp_snow and thr_temp_rain
        accumulation = tf.where(
            state.air_temp <= cfg.modules.smb_accmelt.thr_temp_snow,
            state.precipitation,
            tf.where(
                state.air_temp >= cfg.modules.smb_accmelt.thr_temp_rain,
                0.0,
                state.precipitation
                * (cfg.modules.smb_accmelt.thr_temp_rain - state.air_temp)
                / (cfg.modules.smb_accmelt.thr_temp_rain - cfg.modules.smb_accmelt.thr_temp_snow),
            ),
        )

        # unit to [  kg * m^(-2) * y^(-1) water eq  ] -> [ m water eq / d]
        accumulation /= (accumulation.shape[0] * cfg.modules.smb_accmelt.wat_density) 

        # correct for snow re-distribution
        accumulation *= state.snow_redistribution  # unit to [ m water eq. / d ]

        accumulation *= cfg.modules.smb_accmelt.weight_accumulation  # unit to [ m water eq. / d ]

        pos_temp = tf.where(state.air_temp > 0.0, state.air_temp, 0.0)  # unit is [°C]

        ablation = []  # [ unit : water-eq. m / d ]

        # the snow depth (=0 or >0) is necessary to find what melt factor to apply
        snow_depth = tf.zeros((state.air_temp.shape[1], state.air_temp.shape[2]))
            
        for kk in range(state.air_temp.shape[0]):
                
            # shift to hydro year, i.e. start Oct. 1
            k = (kk+int(state.air_temp.shape[0]*cfg.modules.smb_accmelt.shift_hydro_year))%(state.air_temp.shape[0]) 

            # add accumulation to the snow depth
            snow_depth += accumulation[k]

            # the ablation (unit is m water eq. / d) is the product
            # of positive temp  with melt factors for ice, or snow
            # (Fm + ris * state.direct_radiation has unit [m ice eq. / (°C d)] )
            ablation.append(
                tf.where(
                    snow_depth == 0,
                    pos_temp[k] * (Fm + ri * state.direct_radiation[k]) * 0.91,
                    pos_temp[k] * (Fm + rs * state.direct_radiation[k]) * 0.91,
                )
            )

            ablation[-1] *= cfg.modules.smb_accmelt.weight_ablation

            # remove snow melt to snow depth, and cap it as snow_depth can not be negative
            snow_depth = tf.clip_by_value(snow_depth - ablation[-1], 0.0, 1.0e10)

        # sum accumulation and ablation over the year, and conversion to ice equivalent
        state.smb = tf.math.reduce_sum(accumulation - ablation, axis=0)* (
            cfg.modules.smb_accmelt.wat_density / cfg.modules.smb_accmelt.ice_density
        )

        if hasattr(state,'icemask'):
            state.smb  = tf.where((state.smb<0)|(state.icemask>0.5),state.smb,-10)

        state.tlast_smb.assign(state.t)

        state.tcomp_smb[-1] -= time.time()
        state.tcomp_smb[-1] *= -1


def finalize(cfg,state):
    pass
