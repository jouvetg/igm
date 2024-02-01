#!/usr/bin/env python3

# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time


def params(parser):
    parser.add_argument(
        "--avalanche_update_freq",
        type=float,
        default=1,
        help="Update avalanche each X years (1)",
    )

    parser.add_argument(
        "--avalanche_angleOfRepose",
        type=float,
        default=30,
        help="Angle of repose (30°)",
    )


def initialize(params, state):
    state.tcomp_avalanche = []
    state.tlast_avalanche = tf.Variable(params.time_start, dtype=tf.float32)


def update(params, state):
    if (state.t - state.tlast_avalanche) >= params.avalanche_update_freq:
        if hasattr(state, "logger"):
            state.logger.info("Update AVALANCHE at time : " + str(state.t.numpy()))

        state.tcomp_avalanche.append(time.time())

        H = state.thk
        Zb = state.topg
        Zi = Zb + H
        dHRepose = state.dx * tf.math.tan(
            params.avalanche_angleOfRepose * np.pi / 180.0
        )
        Ho = tf.maximum(H, 0)

        count = 0

        while True:
            count += 1

            dZidx_down = tf.pad(
                tf.maximum(Zi[:, 1:] - Zi[:, :-1], 0), [[0, 0], [1, 0]], "CONSTANT"
            )
            dZidx_up = tf.pad(
                tf.maximum(Zi[:, :-1] - Zi[:, 1:], 0), [[0, 0], [0, 1]], "CONSTANT"
            )
            dZidx = tf.maximum(dZidx_down, dZidx_up)

            dZidy_left = tf.pad(
                tf.maximum(Zi[1:, :] - Zi[:-1, :], 0), [[1, 0], [0, 0]], "CONSTANT"
            )
            dZidy_right = tf.pad(
                tf.maximum(Zi[:-1, :] - Zi[1:, :], 0), [[0, 1], [0, 0]], "CONSTANT"
            )
            dZidy = tf.maximum(dZidy_right, dZidy_left)

            grad = tf.math.sqrt(dZidx**2 + dZidy**2)
            gradT = dZidy_left + dZidy_right + dZidx_down + dZidx_up
            gradT = tf.where(gradT == 0, 1, gradT)
            grad = tf.where(Ho < 0.1, 0, grad)

            mxGrad = tf.reduce_max(grad)
            if mxGrad <= 1.1 * dHRepose:
                break

            delH = tf.maximum(0, (grad - dHRepose) / 3.0)

            Htmp = Ho
            Ho = tf.maximum(0, Htmp - delH)
            delH = Htmp - Ho

            delHup = tf.pad(
                delH[:, :-1] * dZidx_up[:, :-1] / gradT[:, :-1],
                [[0, 0], [1, 0]],
                "CONSTANT",
            )
            delHdn = tf.pad(
                delH[:, 1:] * dZidx_down[:, 1:] / gradT[:, 1:],
                [[0, 0], [0, 1]],
                "CONSTANT",
            )
            delHrt = tf.pad(
                delH[:-1, :] * dZidy_right[:-1, :] / gradT[:-1, :],
                [[1, 0], [0, 0]],
                "CONSTANT",
            )
            delHlt = tf.pad(
                delH[1:, :] * dZidy_left[1:, :] / gradT[1:, :],
                [[0, 1], [0, 0]],
                "CONSTANT",
            )

            Ho = tf.maximum(0, Ho + delHdn + delHup + delHlt + delHrt)

            Zi = Zb + Ho

        # print(count)

        # fig = plt.figure(figsize=(10, 10))
        # plt.imshow( Ho + tf.where(H<0,H,0) - state.thk ,origin='lower'); plt.colorbar()

        state.thk = Ho + tf.where(H < 0, H, 0)

        state.usurf = state.topg + state.thk

        state.tlast_avalanche.assign(state.t)

        state.tcomp_avalanche[-1] -= time.time()
        state.tcomp_avalanche[-1] *= -1


def finalize(params, state):
    pass
