#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file 

import numpy as np
import os
import datetime, time
import tensorflow as tf

from igm.modules.utils import *

def params_write_particles(parser):
    parser.add_argument(
        "--add_topography_to_particles",
        type=str2bool,
        default=False,
        help="Add topg",
    )


def initialize_write_particles(params, state):
    state.tcomp_write_particles = []
 
    directory = os.path.join(params.working_dir, "trajectories")
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)

    os.system(
        "echo rm -r "
        + os.path.join(params.working_dir, "trajectories")
        + " >> clean.sh"
    )

    if params.add_topography_to_particles:
        ftt = os.path.join(params.working_dir, "trajectories", "topg.csv")
        array = tf.transpose(
            tf.stack([state.X[state.X > 0], state.Y[state.X > 0], state.topg[state.X > 0]])
        )
        np.savetxt(ftt, array, delimiter=",", fmt="%.2f", header="x,y,z")


def update_write_particles(params, state):
    if state.saveresult:
        state.tcomp_write_particles.append(time.time())

        f = os.path.join(
            params.working_dir,
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

        ft = os.path.join(params.working_dir, "trajectories", "time.dat")
        with open(ft, "a") as f:
            print(state.t.numpy(), file=f)

        if params.add_topography_to_particles:
            ftt = os.path.join(
                params.working_dir,
                "trajectories",
                "usurf-" + "{:06d}".format(int(state.t.numpy())) + ".csv",
            )
            array = tf.transpose(
                tf.stack(
                    [state.X[state.X > 1], state.Y[state.X > 1], state.usurf[state.X > 1]]
                )
            )
            np.savetxt(ftt, array, delimiter=",", fmt="%.2f", header="x,y,z")

        state.tcomp_write_particles[-1] -= time.time()
        state.tcomp_write_particles[-1] *= -1


def finalize_write_particles(params, state):
    pass
