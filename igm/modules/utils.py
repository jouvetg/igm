#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
This util files provides a large number of usefull function for other modules
"""

import tensorflow as tf

# constrains wildcard imports - not necessary but for security reasons - update if a utility is added...
__all__ = [
    "str2bool",
    "getmag",
    "getmag3d",
    "compute_gradient_tf",
    "compute_divflux",
    "interp1d_tf",
    "complete_data",
    "interpolate_bilinear_tf"
]

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
def getmag3d(u, v):
    """
    return the norm of a 3D vector, e.g. to compute velbase_mag
    """
    return tf.norm(
        tf.concat([tf.expand_dims(u, axis=0), tf.expand_dims(v, axis=0)], axis=0),
        axis=0,
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
def compute_divflux(u, v, h, dx, dy, method='upwind'):
    """
    upwind computation of the divergence of the flux : d(u h)/dx + d(v h)/dy
    First, u and v are computed on the staggered grid (i.e. cell edges)
    Second, one extend h horizontally by a cell layer on any bords (assuming same value)
    Third, one compute the flux on the staggered grid slecting upwind quantities
    Last, computing the divergence on the staggered grid yields values def on the original grid
    """

    if method == 'upwind':

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

    elif method == 'centered':

        Qx = u * h  
        Qy = v * h  
        
        Qx = tf.concat(
            [Qx[:, 0:1], 0.5 * (Qx[:, :-1] + Qx[:, 1:]), Qx[:, -1:]], 1
        )  # has shape (ny,nx+1) 
        
        Qy = tf.concat(
            [Qy[0:1, :], 0.5 * (Qy[:-1, :] + Qy[1:, :]), Qy[-1:, :]], 0
        )  # has shape (ny+1,nx)
        
        ## Computation of the divergence, final shape is (ny,nx)
    return (Qx[:, 1:] - Qx[:, :-1]) / dx + (Qy[1:, :] - Qy[:-1, :]) / dy

 
def minmod(a, b):
    return tf.where( (tf.abs(a)<tf.abs(b))&(a*b>0.0), a, tf.where((tf.abs(a)>tf.abs(b))&(a*b>0.0),b,0))
    
def maxmod(a, b):
    return tf.where( (tf.abs(a)<tf.abs(b))&(a*b>0.0), b, tf.where((tf.abs(a)>tf.abs(b))&(a*b>0.0),a,0))

@tf.function()
def compute_divflux_slope_limiter(u, v, h, dx, dy, dt, slope_type):
    """
    upwind computation of the divergence of the flux : d(u h)/dx + d(v h)/dy
    propose a slope limiter for the upwind scheme with 3 options : godunov, minmod, superbee
    
    References :
    - Numerical Methods for Engineers, Leif Rune Hellevik, book
      https://folk.ntnu.no/leifh/teaching/tkt4140/._main074.html
    
    - hydro_examples github page, Michael Zingale, Ian Hawke
     collection of simple python codes that demonstrate some basic techniques used in hydrodynamics codes.
     https://github.com/python-hydro/hydro_examples
    """
    
    u = tf.concat( [u[:, 0:1], 0.5 * (u[:, :-1] + u[:, 1:]), u[:, -1:]], 1 )  # has shape (ny,nx+1)
    v = tf.concat( [v[0:1, :], 0.5 * (v[:-1, :] + v[1:, :]), v[-1:, :]], 0 )  # has shape (ny+1,nx)

    Hx = tf.pad(h, [[0,0],[2,2]], 'CONSTANT') # (ny,nx+4)
    Hy = tf.pad(h, [[2,2],[0,0]], 'CONSTANT') # (ny+4,nx)
    
    sigpx = (Hx[:,2:]-Hx[:,1:-1])/dx    # (ny,nx+2)
    sigmx = (Hx[:,1:-1]-Hx[:,:-2])/dx   # (ny,nx+2) 

    sigpy = (Hy[2:,:] -Hy[1:-1,:])/dy   # (ny+2,nx)
    sigmy = (Hy[1:-1,:]-Hy[:-2,:])/dy   # (ny+2,nx) 

    if slope_type == "godunov":
 
        slopex = tf.zeros_like(sigpx)  
        slopey = tf.zeros_like(sigpy)  
        
    elif slope_type == "minmod":
 
        slopex  = minmod(sigmx,sigpx) 
        slopey  = minmod(sigmy,sigpy)

    elif slope_type == "superbee":

        sig1x  = minmod( sigpx , 2.0*sigmx )
        sig2x  = minmod( sigmx , 2.0*sigpx )
        slopex = maxmod( sig1x, sig2x)

        sig1y  = minmod( sigpy , 2.0*sigmy )
        sig2y  = minmod( sigmy , 2.0*sigpy )
        slopey = maxmod( sig1y, sig2y)

    w   = Hx[:,1:-2] + 0.5*dx*(1.0 - u*dt/dx)*slopex[:,:-1]      #  (ny,nx+1)      
    e   = Hx[:,2:-1] - 0.5*dx*(1.0 + u*dt/dx)*slopex[:,1:]       #  (ny,nx+1)    
    
    s   = Hy[1:-2,:] + 0.5*dy*(1.0 - v*dt/dy)*slopey[:-1,:]      #  (ny+1,nx)      
    n   = Hy[2:-1,:] - 0.5*dy*(1.0 + v*dt/dy)*slopey[1:,:]       #  (ny+1,nx)    
     
    Qx = u * tf.where(u > 0, w, e)  #  (ny,nx+1)   
    Qy = v * tf.where(v > 0, s, n)  #  (ny+1,nx)   
     
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


def complete_data(state):
    """
    This function adds a postriori import fields such as X, Y, x, dx, ....
    """

    # define grids, i.e. state.X and state.Y has same shape as state.thk
    if not hasattr(state, "X"):
        state.X, state.Y = tf.meshgrid(state.x, state.y)

    # define cell spacing
    if not hasattr(state, "dx"):
        state.dx = state.x[1] - state.x[0]

    # define dX
    if not hasattr(state, "dX"):
        state.dX = tf.ones_like(state.X) * state.dx       
    
    # if thickness is not defined in the netcdf, then it is set to zero
    if not hasattr(state, "thk"):
        state.thk = tf.Variable(tf.zeros((state.y.shape[0], state.x.shape[0])))
        
    assert hasattr(state, "topg") | hasattr(state, "usurf")
    
    # case usurf defined, topg is not defined
    if not hasattr(state, "topg"):
        state.topg = tf.Variable(state.usurf - state.thk) 

    # case usurf not defined, topg is defined
    if not hasattr(state, "usurf"): 
        state.usurf = tf.Variable(state.topg + state.thk) 

@tf.function
def interpolate_bilinear_tf(
    grid: tf.Tensor,
    query_points: tf.Tensor,
    indexing: str = "ij",
) -> tf.Tensor:
    """

    This function originally comes from tensorflow-addon library
    (https://www.tensorflow.org/addons/api_docs/python/tfa/image/interpolate_bilinear)
    but the later was deprecated, therefore we copied the function here to avoid
    being dependent on a deprecated library.

    Similar to Matlab's interp2 function.

    Finds values for query points on a grid using bilinear interpolation.

    Args:
      grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
      query_points: a 3-D float `Tensor` of N points with shape
        `[batch, N, 2]`.
      indexing: whether the query points are specified as row and column (ij),
        or Cartesian coordinates (xy).

    Returns:
      values: a 3-D `Tensor` with shape `[batch, N, channels]`

    Raises:
      ValueError: if the indexing mode is invalid, or if the shape of the
        inputs invalid.
    """

    grid = tf.convert_to_tensor(grid)
    query_points = tf.convert_to_tensor(query_points)
    grid_shape = tf.shape(grid)
    query_shape = tf.shape(query_points)

    with tf.name_scope("interpolate_bilinear"):
        grid_shape = tf.shape(grid)
        query_shape = tf.shape(query_points)

        batch_size, height, width, channels = (
            grid_shape[0],
            grid_shape[1],
            grid_shape[2],
            grid_shape[3],
        )

        num_queries = query_shape[1]

        query_type = query_points.dtype
        grid_type = grid.dtype

        alphas = []
        floors = []
        ceils = []
        index_order = [0, 1] if indexing == "ij" else [1, 0]
        unstacked_query_points = tf.unstack(query_points, axis=2, num=2)

        for i, dim in enumerate(index_order):
            with tf.name_scope("dim-" + str(dim)):
                queries = unstacked_query_points[dim]

                size_in_indexing_dimension = grid_shape[i + 1]

                # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
                # is still a valid index into the grid.
                max_floor = tf.cast(size_in_indexing_dimension - 2, query_type)
                min_floor = tf.constant(0.0, dtype=query_type)
                floor = tf.math.minimum(
                    tf.math.maximum(min_floor, tf.math.floor(queries)), max_floor
                )
                int_floor = tf.cast(floor, tf.dtypes.int32)
                floors.append(int_floor)
                ceil = int_floor + 1
                ceils.append(ceil)

                # alpha has the same type as the grid, as we will directly use alpha
                # when taking linear combinations of pixel values from the image.
                alpha = tf.cast(queries - floor, grid_type)
                min_alpha = tf.constant(0.0, dtype=grid_type)
                max_alpha = tf.constant(1.0, dtype=grid_type)
                alpha = tf.math.minimum(tf.math.maximum(min_alpha, alpha), max_alpha)

                # Expand alpha to [b, n, 1] so we can use broadcasting
                # (since the alpha values don't depend on the channel).
                alpha = tf.expand_dims(alpha, 2)
                alphas.append(alpha)

            flattened_grid = tf.reshape(grid, [batch_size * height * width, channels])
            batch_offsets = tf.reshape(
                tf.range(batch_size) * height * width, [batch_size, 1]
            )

        # This wraps tf.gather. We reshape the image data such that the
        # batch, y, and x coordinates are pulled into the first dimension.
        # Then we gather. Finally, we reshape the output back. It's possible this
        # code would be made simpler by using tf.gather_nd.
        def gather(y_coords, x_coords, name):
            with tf.name_scope("gather-" + name):
                linear_coordinates = batch_offsets + y_coords * width + x_coords
                gathered_values = tf.gather(flattened_grid, linear_coordinates)
                return tf.reshape(gathered_values, [batch_size, num_queries, channels])

        # grab the pixel values in the 4 corners around each query point
        top_left = gather(floors[0], floors[1], "top_left")
        top_right = gather(floors[0], ceils[1], "top_right")
        bottom_left = gather(ceils[0], floors[1], "bottom_left")
        bottom_right = gather(ceils[0], ceils[1], "bottom_right")

        # now, do the actual interpolation
        with tf.name_scope("interpolate"):
            interp_top = alphas[1] * (top_right - top_left) + top_left
            interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
            interp = alphas[0] * (interp_bottom - interp_top) + interp_top

        return interp
