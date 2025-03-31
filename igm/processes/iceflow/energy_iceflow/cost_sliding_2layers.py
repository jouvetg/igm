#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from .utils import stag4

@tf.function()
def cost_sliding_2layers(U, V, slidingco, exp_weertman, regu_weertman):
         
#    sloptopgx, sloptopgy = compute_gradient_stag(usurf - thksurf, dX, dX)

    s = 1.0 + 1.0 / exp_weertman
    C = 1.0 * slidingco

    # C_slid is unit Mpa y^m m^(-m) * m^(1+m) * y^(-1-m)  = Mpa  m/y
    N = (
        stag4(U[:, 0, :, :] ** 2 + V[:, 0, :, :] ** 2)
        + regu_weertman**2
#        + (stag4(U[:, 0, :, :]) * sloptopgx + stag4(V[:, 0, :, :]) * sloptopgy) ** 2
    )
    C_slid = stag4(C) * N ** (s / 2) / s

    return C_slid
