#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os
import time
import tensorflow as tf
import shutil
from igm.modules.utils import *


def params(parser):
    parser.add_argument(
        "--wpar_add_topography",
        type=str2bool,
        default=False,
        help="Add topg",
    )


def initialize(params, state):
    state.tcomp_write_particles = []

    directory = "trajectories"
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)

    os.system( "echo rm -r " + "trajectories" + " >> clean.sh" )

    if params.wpar_add_topography:
        ftt = os.path.join("trajectories", "topg.csv")
        array = tf.transpose(
            tf.stack(
                [state.X[state.X > 0], state.Y[state.X > 0], state.topg[state.X > 0]]
            )
        )
        np.savetxt(ftt, array, delimiter=",", fmt="%.2f", header="x,y,z")


def update(params, state):
    if state.saveresult:
        state.tcomp_write_particles.append(time.time())

        f = os.path.join(
            "trajectories",
            "traj-" + "{:06d}".format(int(state.t.numpy())) + ".csv",
        )

        ID = tf.cast(tf.range(state.xpos.shape[0]), dtype="float32")
        array = tf.transpose(
            tf.stack(
                [
                    ID,
                    state.xpos,
                    state.ypos,
                    state.zpos,
                    state.rhpos,
                    state.tpos,
                    state.englt,
                ],
                axis=0,
            )
        )
        np.savetxt(f, array, delimiter=",", fmt="%.2f", header="Id,x,y,z,rh,t,englt")

        ft = os.path.join("trajectories", "time.dat")
        with open(ft, "a") as f:
            print(state.t.numpy(), file=f)

        if params.wpar_add_topography:
            ftt = os.path.join(
                "trajectories",
                "usurf-" + "{:06d}".format(int(state.t.numpy())) + ".csv",
            )
            array = tf.transpose(
                tf.stack(
                    [
                        state.X[state.X > 1],
                        state.Y[state.X > 1],
                        state.usurf[state.X > 1],
                    ]
                )
            )
            np.savetxt(ftt, array, delimiter=",", fmt="%.2f", header="x,y,z")

        state.tcomp_write_particles[-1] -= time.time()
        state.tcomp_write_particles[-1] *= -1


def finalize(params, state):
    pass
