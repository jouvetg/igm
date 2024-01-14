#!/usr/bin/env python3

# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time


def params_gflex(parser):
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


def initialize_gflex(params, state):
    import gflex
    from gflex.f2d import F2D

    state.tcomp_gflex = []
    state.tlast_gflex = tf.Variable(params.time_start, dtype=tf.float32)

    state.flex = F2D()

    state.flex.giafreq = params.gflex_update_freq
    state.flex.giatime = params.gflex_update_freq
    state.flex.Quiet = False
    state.flex.Method = "FD"
    state.flex.PlateSolutionType = "vWC1994"
    state.flex.Solver = "direct"
    state.flex.g = 9.81
    state.flex.E = 100e9
    state.flex.nu = 0.25
    state.flex.rho_m = 3300.0
    state.flex.rho_fill = 0
    state.flex.dx = state.dx.numpy()
    state.flex.dy = state.dx.numpy()
    state.flex.BC_W = "Periodic"
    state.flex.BC_E = "Periodic"
    state.flex.BC_S = "Periodic"
    state.flex.BC_N = "Periodic"

    if not hasattr(state, "Te"):
        state.flex.Te = np.ones_like(state.thk.numpy() * params.gflex_default_Te)


def update_gflex(params, state):
    import gflex
    from scipy.interpolate import griddata
    
    def downsample_array_to_resolution(arr, dx, target_resolution):
        """
        Downsample a 2D array to a specified resolution using bilinear interpolation (chatgpt).

        """
        m, n = arr.shape
        x = np.arange(0, n) * dx
        y = np.arange(0, m) * dx

        target_x = np.arange(0, n, target_resolution / dx) * dx
        target_y = np.arange(0, m, target_resolution / dx) * dx

        xx, yy = np.meshgrid(x, y)
        target_xx, target_yy = np.meshgrid(target_x, target_y)
        target_xx = target_xx.astype(np.int32)
        target_yy = target_yy.astype(np.int32)

        points = np.column_stack((xx.flatten(), yy.flatten()))
        target_points = np.column_stack((target_xx.flatten(), target_yy.flatten()))

        downsampled_array = griddata(points, arr.flatten(), target_points, method='linear')
        return downsampled_array.reshape(len(target_y), len(target_x)), target_points, points, x, y

    def upsample_result_to_original_resolution(result, target_points, points, x, y, original_resolution):
        """
        Upsample a 2D array to the original resolution using bilinear interpolation (chatgpt).

        """
        upsampled_result = griddata(target_points, result.flatten(), points, method='linear')
        return upsampled_result.reshape(len(y), len(x))

    if (state.t - state.tlast_gflex) >= params.gflex_update_freq:
        if hasattr(state, "logger"):
            state.logger.info("Update gflex at time : " + str(state.t.numpy()))

        state.tcomp_gflex.append(time.time())

        state.flex.Te = state.Te # Elastic thickness [m] -- scalar or array
        state.flex.Te = state.flex.Te.numpy()        
        state.flex.qs = state.thk.numpy() * 917 * 9.81  # Populating this template
        
        if state.dx < 1000:
            state.flex.Te, target_points,points, x, y = downsample_array_to_resolution(state.flex.Te, state.flex.dx, 1000)
            state.flex.qs, target_points,points, x, y = downsample_array_to_resolution(state.flex.qs, state.flex.dx, 1000)
            state.flex.initialize()
            state.flex.run()
            state.flex.finalize()
            state.flex.w = upsample_result_to_original_resolution(state.flex.w, target_points, points, x, y, state.flex.dx)
        else:
            state.flex.initialize()
            state.flex.run()
            state.flex.finalize()
        state.topg = state.topg + state.flex.w
        state.usurf = state.topg + state.thk
        # state.flex.output()

        state.tlast_gflex.assign(state.t)

        state.tcomp_gflex[-1] -= time.time()
        state.tcomp_gflex[-1] *= -1


def finalize_gflex(params, state):
    pass
