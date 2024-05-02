#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file 

import tensorflow as tf
import numpy as np
from scipy.interpolate import interp1d

def params(parser):

    parser.add_argument(
        "--pdela",
        type=float,
        default=30000,
        help="present-day ELA",
    )
    parser.add_argument(
        "--deladt",
        type=float,
        default=200,
        help="Delta dt",
    )
    parser.add_argument(
        "--gradabl",
        type=float,
        default=0.0067,
        help="Abalation gradient",
    )
    parser.add_argument(
        "--gradacc",
        type=float,
        default=0.0005,
        help="Accumulation gradient",
    )
    parser.add_argument(
        "--maxacc",
        type=float,
        default=1.0,
        help="Maximum accumulation",
    )

def initialize(params,state):
    """
        Retrieve the Temperature difference from the EPICA signal
    """
    # load the EPICA signal from theparams,state official data
    ss = np.loadtxt('data/EDC_dD_temp_estim.tab',dtype=np.float32,skiprows=31)
    time = ss[:,1] * -1000  # extract time BP, chnage unit to yeat
    dT   = ss[:,3]          # extract the dT, i.e. global temp. difference
    state.dT =  interp1d(time,dT, fill_value=(dT[0], dT[-1]),bounds_error=False)

def update(params,state):
    """
        mass balance 'signal'
    """

    # define ELA as function of EPICA's Delta T, ELA's present day (pdela) and Dela/Dt (deladt)
    ela     = params.pdela + params.deladt*state.dT(state.t) # for rhine

    # that's SMB param with ELA, ablation and accc gradient, and max accumulation
    # i.e. SMB =       gradabl*(z-ela)           if z<ela, 
    #          =  min( gradacc*(z-ela) , maxacc) if z>ela.
    state.smb  = state.usurf - ela
    state.smb  *= tf.where(tf.less(state.smb , 0), params.gradabl, params.gradacc)
    state.smb  = tf.clip_by_value(state.smb , -100, params.maxacc)

def finalize(params, state):
    pass
