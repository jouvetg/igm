#!/usr/bin/env python3

# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from igm.modules.utils import *



def params(parser):
    parser.add_argument(
        "--gflex_update_freq",
        type=float,
        default=100.0,
        help="Update gflex each X years (1)",
    )
    parser.add_argument(
        "--gflex_default_Te",
        type=float,
        default=50000,
        help="Default value for Te (Elastic thickness [m]) if not given as ncdf file",
    )
    parser.add_argument(
        "--gflex_dx",
        type=float,
        default=1000,
        help="Default resolution for computing isostasy (m).",
    )
    parser.add_argument(
        "--gflex_pad",
        type=str2bool,
        default=False,
        help="Default padding option",
    )
    parser.add_argument(
        "--gflex_quiet",
        type=str2bool,
        default=True,
        help="Default padding option",
    )

def initialize(params, state):
    from gflex.f2d import F2D

    if not hasattr(state,"tcomp_gflex"):
        state.tcomp_gflex = []
        state.tlast_gflex = tf.Variable(params.time_start, dtype=tf.float32)
        state.topg0 = state.usurf - state.thk

    state.flex = F2D()

    state.flex.giafreq = params.gflex_update_freq
    state.flex.giatime = params.gflex_update_freq
    state.flex.dx = params.gflex_dx
    state.flex.Quiet = False
    state.flex.pad = params.gflex_pad
    state.flex.Method = "FD"
    state.flex.PlateSolutionType = "vWC1994"
    state.flex.Solver = "direct"
    state.flex.g = 9.81
    state.flex.E = 100e9
    state.flex.nu = 0.25
    state.flex.rho_m = 3300.0
    state.flex.rho_fill = 0
    state.flex.dy = state.flex.dx
    state.flex.BC_W = "0Displacement0Slope"
    state.flex.BC_E = "0Displacement0Slope"
    state.flex.BC_S = "0Displacement0Slope"
    state.flex.BC_N = "0Displacement0Slope"

    if not hasattr(state, "Te"):
        state.flex.Te0 = np.ones_like(state.thk.numpy()) * params.gflex_default_Te
    else:
        state.flex.Te0 = state.Te    
    if not hasattr(state,"tcomp_gflex"):
        state.flex.Te0 = state.flex.Te   
    

def update(params, state):
    from scipy.interpolate import griddata
    
    initialize(params, state)
    
    def downsample_array_to_resolution(arr, dx, target_resolution):
        """
        Downsample a 2D array to a specified resolution using bilinear interpolation (chatgpt).

        """
        m, n = arr.shape
        x = np.arange(0, n) * dx
        y = np.arange(0, m) * dx
        xx, yy = np.meshgrid(x, y)

        target_x = np.arange(0, n, target_resolution / dx) * dx
        target_y = np.arange(0, m, target_resolution / dx) * dx
        target_xx, target_yy = np.meshgrid(target_x, target_y)
        
        points = np.column_stack((xx.flatten(), yy.flatten()))
        target_points = np.column_stack((target_xx.flatten(), target_yy.flatten()))

        downsampled_array = griddata(points, arr.flatten(), target_points, method='nearest')
        return downsampled_array.reshape(len(target_y), len(target_x))
      
    def pad_arrays(params,state):
        """
        Pad Te and load arrays with one flexural wavelenth based on the mean effective elastic thickness. This is to avoid boundary effects.
        """
        mean_Te = np.mean(state.flex.Te, where=~np.isnan(state.flex.Te))
        fw = 2 * np.pi * ((state.flex.E * mean_Te**3) / (12 * (1 - state.flex.nu**2) * (state.flex.rho_m - 917) * 9.81))**0.25
        pad_width = round(fw / state.flex.dx)
        if np.shape(state.flex.Te)==np.shape(state.flex.qs):
            state.flex.Te = np.pad(state.flex.Te, pad_width, mode='constant', constant_values=mean_Te)
        else:
            rp, cp = np.shape(state.flex.Te) # dims of padded grid
            pad_width = round((rp-r)/2)
        state.flex.qs = np.pad(state.flex.qs, pad_width, mode='constant', constant_values=0)
        
        return pad_width

    if (state.t - state.tlast_gflex) >= params.gflex_update_freq:
        if hasattr(state, "logger"):
            state.logger.info("Update gflex at time : " + str(state.t.numpy()))

        state.tcomp_gflex.append(time.time())

        state.flex.Te = np.float32(state.flex.Te0)       
        state.flex.qs = state.thk.numpy() * 917 * 9.81  # convert thicknesses to loads
        
        if state.dx < state.flex.dx:
            if np.shape(state.flex.Te)==np.shape(state.flex.qs): # in case you want to use a pre-padded (and resampled) Te grid
                state.flex.Te = downsample_array_to_resolution(state.flex.Te, state.dx, state.flex.dx)  
            state.flex.qs = downsample_array_to_resolution(state.flex.qs, state.dx, state.flex.dx)    
            
            r, c = np.shape(state.flex.qs) # dimension of the downsampled arrays
            rr, cc = np.shape( state.thk.numpy()) # dimension of the original array
            
            if state.flex.pad == True:
                p = pad_arrays(params,state) # padding of one flexural wavelength
            else:
                p = 0
            
            # gFlex
            state.flex.initialize()
            state.flex.run()
            state.flex.finalize()
            
            # transform result back to original resolution and dimension
            u, v = np.shape(state.flex.w) # dimension of the padded arrays
            start_row = p
            end_row = start_row + r
            start_col = p
            end_col = start_col + c
            x = np.arange(0, v)
            y = np.arange(0, u)
            X, Y = np.meshgrid(x, y)
            points = np.column_stack((X.flatten(), Y.flatten()))
            values = state.flex.w.flatten()
            target_x = np.linspace(start_col, end_col, cc)
            target_y = np.linspace(start_row, end_row, rr)
            target_X, target_Y = np.meshgrid(target_x, target_y)
                        
            state.flex.w = griddata(points, values, (target_X, target_Y), method='linear', fill_value = 0)
        else:
            if state.flex.pad == True:
                p = pad_arrays(params,state)
            
            state.flex.initialize()
            state.flex.run()
            state.flex.finalize()
            
            if state.flex.pad == True:
                # remove the padded cols and rows
                state.flex.w = state.flex.w[p:-p,p:-p]
        
        # add the deflection to the topography 
        state.topg = state.topg0 + state.flex.w
        state.usurf = state.topg + state.thk
        # state.flex.plotChoice='both'
        # state.flex.output()
        # plt.imshow(state.flex.w)
        # plt.colorbar()

        state.tlast_gflex.assign(state.t)

        state.tcomp_gflex[-1] -= time.time()
        state.tcomp_gflex[-1] *= -1


def finalize(params, state):
    pass
