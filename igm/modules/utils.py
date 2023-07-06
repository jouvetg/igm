#!/usr/bin/env python3

"""
Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
Published under the GNU GPL (Version 3), check at the LICENSE file
"""

import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import tensorflow as tf


def str2bool(v):
    return v.lower() in ("true", "1")


@tf.function()
def getmag(u, v):
    """
    return the norm of a 2D vector, e.g. to compute velbase_mag
    """
    return tf.norm(
        tf.concat([tf.expand_dims(u, axis=-1), tf.expand_dims(v, axis=-1)], axis=2),
        axis=2,
    )


@tf.function()
def compute_gradient_tf(s, dx, dy):
    """
    compute spatial 2D gradient of a given field
    """

    # EX = tf.concat([s[:, 0:1], 0.5 * (s[:, :-1] + s[:, 1:]), s[:, -1:]], 1)
    # diffx = (EX[:, 1:] - EX[:, :-1]) / dx

    # EY = tf.concat([s[0:1, :], 0.5 * (s[:-1, :] + s[1:, :]), s[-1:, :]], 0)
    # diffy = (EY[1:, :] - EY[:-1, :]) / dy

    EX = tf.concat(
        [
            1.5 * s[:, 0:1] - 0.5 * s[:, 1:2],
            0.5 * s[:, :-1] + 0.5 * s[:, 1:],
            1.5 * s[:, -1:] - 0.5 * s[:, -2:-1],
        ],
        1,
    )
    diffx = (EX[:, 1:] - EX[:, :-1]) / dx

    EY = tf.concat(
        [
            1.5 * s[0:1, :] - 0.5 * s[1:2, :],
            0.5 * s[:-1, :] + 0.5 * s[1:, :],
            1.5 * s[-1:, :] - 0.5 * s[-2:-1, :],
        ],
        0,
    )
    diffy = (EY[1:, :] - EY[:-1, :]) / dy

    return diffx, diffy


@tf.function()
def compute_divflux(u, v, h, dx, dy):
    """
    upwind computation of the divergence of the flux : d(u h)/dx + d(v h)/dy
    First, u and v are computed on the staggered grid (i.e. cell edges)
    Second, one extend h horizontally by a cell layer on any bords (assuming same value)
    Third, one compute the flux on the staggered grid slecting upwind quantities
    Last, computing the divergence on the staggered grid yields values def on the original grid
    """

    ## Compute u and v on the staggered grid
    u = tf.concat(
        [u[:, 0:1], 0.5 * (u[:, :-1] + u[:, 1:]), u[:, -1:]], 1
    )  # has shape (ny,nx+1)
    v = tf.concat(
        [v[0:1, :], 0.5 * (v[:-1, :] + v[1:, :]), v[-1:, :]], 0
    )  # has shape (ny+1,nx)

    # Extend h with constant value at the domain boundaries
    Hx = tf.pad(h, [[0, 0], [1, 1]], "CONSTANT")  # has shape (ny,nx+2)
    Hy = tf.pad(h, [[1, 1], [0, 0]], "CONSTANT")  # has shape (ny+2,nx)

    ## Compute fluxes by selcting the upwind quantities
    Qx = u * tf.where(u > 0, Hx[:, :-1], Hx[:, 1:])  # has shape (ny,nx+1)
    Qy = v * tf.where(v > 0, Hy[:-1, :], Hy[1:, :])  # has shape (ny+1,nx)

    ## Computation of the divergence, final shape is (ny,nx)
    return (Qx[:, 1:] - Qx[:, :-1]) / dx + (Qy[1:, :] - Qy[:-1, :]) / dy


@tf.function()
def interp1d_tf(xs, ys, x):
    """
    This is a 1D interpolation tensorflow implementation
    """
    x = tf.clip_by_value(x, tf.reduce_min(xs), tf.reduce_max(xs))

    # determine the output data type
    ys = tf.convert_to_tensor(ys)
    dtype = ys.dtype

    # normalize data types
    ys = tf.cast(ys, tf.float64)
    xs = tf.cast(xs, tf.float64)
    x = tf.cast(x, tf.float64)

    # pad control points for extrapolation
    xs = tf.concat([[xs.dtype.min], xs, [xs.dtype.max]], axis=0)
    ys = tf.concat([ys[:1], ys, ys[-1:]], axis=0)

    # compute slopes, pad at the edges to flatten
    ms = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
    ms = tf.pad(ms[:-1], [(1, 1)])

    # solve for intercepts
    bs = ys - ms * xs

    # search for the line parameters at each input data point
    # create a grid of the inputs and piece breakpoints for thresholding
    # rely on argmax stopping on the first true when there are duplicates,
    # which gives us an index into the parameter vectors
    i = tf.math.argmax(xs[..., tf.newaxis, :] > x[..., tf.newaxis], axis=-1)
    m = tf.gather(ms, i, axis=-1)
    b = tf.gather(bs, i, axis=-1)

    # apply the linear mapping at each input data point
    y = m * x + b
    return tf.cast(tf.reshape(y, tf.shape(x)), dtype)


def complete_data(self):
    """
    This function adds a postriori import fields such as X, Y, x, dx, ....
    """

    # define grids, i.e. self.X and self.Y has same shape as self.thk
    if not hasattr(self, "X"):
        self.X, self.Y = tf.meshgrid(self.x, self.y)

    # define cell spacing
    if not hasattr(self, "dx"):
        self.dx = self.x[1] - self.x[0]

    # define dX
    if not hasattr(self, "dX"):
        self.dX = tf.ones_like(self.X) * self.dx

    # if thickness is not defined in the netcdf, then it is set to zero
    if not hasattr(self, "thk"):
        self.thk = tf.Variable(tf.zeros((self.y.shape[0], self.x.shape[0])))

    # at this point, we should have defined at least topg or usurf
    assert hasattr(self, "topg") | hasattr(self, "usurf")

    # define usurf (or topg) from topg (or usurf) and thk
    if hasattr(self, "usurf"):
        self.topg = tf.Variable(self.usurf - self.thk)
    else:
        self.usurf = tf.Variable(self.topg + self.thk)


def anim_3d_from_ncdf_ex(params):
    """
    Produce a 3d animated Plot using mayavi library from the ex.nc netcdf file
    """
    from mayavi import mlab
    import xarray as xr
    import numpy as np
    import os

    ds = xr.open_dataset(os.path.join(params.working_dir, "ex.nc"), engine="netcdf4")

    X, Y = np.meshgrid(ds.x, ds.y)

    TIME = np.array(ds.time)

    vmin = np.min(ds.velsurf_mag)
    vmax = np.max(ds.velsurf_mag)

    # mlab.figure(bgcolor=(0.16, 0.28, 0.46))

    XX = np.where(ds.thk[0] == 0, np.nan, X)
    YY = np.where(ds.thk[0] == 0, np.nan, Y)
    ZZ = np.where(ds.thk[0] == 0, np.nan, ds.usurf[0])
    CC = np.array(ds.velsurf_mag[0])

    base = mlab.mesh(X, Y, ds.topg[0], colormap="terrain", opacity=0.75)
    surf = mlab.mesh(XX, YY, ZZ, scalars=CC, colormap="jet", vmin=vmin, vmax=vmax)
    mlab.colorbar(surf, orientation="vertical", title="Ice speed (m/y)")
    mlab.title(str(TIME[0]) + " y", size=0.5)
    
    if (hasattr(ds,'uvelsurf')&hasattr(ds,'vvelsurf')&hasattr(ds,'wvelsurf')):
        quiv = mlab.quiver3d(tf.expand_dims(X,axis=0),
                             tf.expand_dims(Y,axis=0),
                             tf.expand_dims(ds.usurf[0],axis=0),
                             tf.expand_dims(ds.uvelsurf[0],axis=0),
                             tf.expand_dims(ds.vvelsurf[0],axis=0),
                             tf.expand_dims(ds.wvelsurf[0],axis=0)
                             )

    @mlab.animate(ui=True)
    def anim():
        for i in range(0, ds.thk.shape[0]):
            surf.mlab_source.z = np.where(ds.thk[i] == 0, np.nan, ds.usurf[i])
            surf.mlab_source.scalars = np.array(ds.velsurf_mag[i])
            mlab.title("Time " + str(int(TIME[i])), size=0.5)
            yield

    anim()
    mlab.show()


def anim_mp4_from_ncdf_ex(params):
    """
    Produce an animated video from the netcdf ex.nc file
    """
    import xarray as xr
    from matplotlib import animation

    ds = xr.open_dataset(os.path.join(params.working_dir, "ex.nc"), engine="netcdf4")

    tas = ds.thk

    # Get a handle on the figure and the axes
    fig, ax = plt.subplots(figsize=(7,7))

    # Plot the initial frame.
    cax = tas[0,:,:].where(tas[0,:,:]>0).plot(
        add_colorbar=True,
        cmap="jet",
        vmin=0,
        vmax=np.max(tas),
        cbar_kwargs={"extend": "neither"}
    )
    
    cax.axes.set_aspect('equal')

    ax.axis("off")

    # Next we need to create a function that updates the values for the colormesh, as well as the title.
    def animate(frame):
        cax.set_array(tas[frame,:,:].where(tas[frame,:,:]>0).values.flatten())
        ax.set_title("Time = " + str(tas.coords["time"].values[frame])[:13])

    # Finally, we use the animation module to create the animation.
    ani = animation.FuncAnimation(
        fig,  # figure
        animate,  # name of the function above
        frames=tas.shape[0],  # Could also be iterable or list
        interval=500,  # ms between frames
    )

    ani.save(os.path.join(params.working_dir, "animation.mp4"))
    
    os.system(
        "echo rm "
        + os.path.join(params.working_dir, "animation.mp4")
        + " >> clean.sh"
    )

