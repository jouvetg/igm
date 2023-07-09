#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file 

"""
This IGM module writes particle time-position in csv files at a given freq.
==============================================================================
Input: self.xpos, self.ypos, self.zpos, self.rhpos, self.tpos, self.englt
Output: csv file in forder trajectories
"""

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


def init_write_particles(params, self):
    self.tcomp["write_particles"] = []
 
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
            tf.stack([self.X[self.X > 0], self.Y[self.X > 0], self.topg[self.X > 0]])
        )
        np.savetxt(ftt, array, delimiter=",", fmt="%.2f", header="x,y,z")


def update_write_particles(params, self):
    if self.saveresult:
        self.tcomp["write_particles"].append(time.time())

        f = os.path.join(
            params.working_dir,
            "trajectories",
            "traj-" + "{:06d}".format(int(self.t.numpy())) + ".csv",
        )

        ID = tf.cast(tf.range(self.xpos.shape[0]), dtype="float32")
        array = tf.transpose(
            tf.stack(
                [
                    ID,
                    self.xpos,
                    self.ypos,
                    self.zpos,
                    self.rhpos,
                    self.tpos,
                    self.englt,
                ],
                axis=0,
            )
        )
        np.savetxt(f, array, delimiter=",", fmt="%.2f", header="Id,x,y,z,rh,t,englt")

        ft = os.path.join(params.working_dir, "trajectories", "time.dat")
        with open(ft, "a") as f:
            print(self.t.numpy(), file=f)

        if params.add_topography_to_particles:
            ftt = os.path.join(
                params.working_dir,
                "trajectories",
                "usurf-" + "{:06d}".format(int(self.t.numpy())) + ".csv",
            )
            array = tf.transpose(
                tf.stack(
                    [self.X[self.X > 1], self.Y[self.X > 1], self.usurf[self.X > 1]]
                )
            )
            np.savetxt(ftt, array, delimiter=",", fmt="%.2f", header="x,y,z")

        self.tcomp["write_particles"][-1] -= time.time()
        self.tcomp["write_particles"][-1] *= -1


def final_write_particles(params, self):
    pass
