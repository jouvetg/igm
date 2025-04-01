import tensorflow as tf
from numba import cuda
import cupy as cp
from cupyx.scipy.interpolate import interpn
import nvtx



def srange(message, color):
    tf.test.experimental.sync_devices()
    return nvtx.start_range(message, color)

def erange(rng):
    tf.test.experimental.sync_devices()
    nvtx.end_range(rng)
    
def interpolate_2d(grid, x, y):
    # https://en.wikipedia.org/wiki/Bilinear_interpolation
    # Note, I had to make some adjustments to fix the CS convention:
    # y is the row axis that starts at the top with index 0 and x is the row axis that stats at the left with index 0
    # So, this works but the notation is slightly different compared to the wiki page (notably when calculating P)
    
    x_1 = tf.floor(x) # left x coordinate
    y_1 = tf.floor(y) # bottom y coordinate
    x_2 = tf.math.ceil(x) # right x coordinate
    y_2 = tf.math.ceil(y) # top y coordinate

    x_1 = tf.cast(x_1, tf.int32)
    y_1 = tf.cast(y_1, tf.int32)
    x_2 = tf.cast(x_2, tf.int32)
    y_2 = tf.cast(y_2, tf.int32)
    
    # Q_12 = tf.gather
    
    # Q_12 = grid[y_1, x_1] # top left corner
    # Q_22 = grid[y_1, x_2] # top right corner
    # Q_11 = grid[y_2, x_1] # bottom left corner
    # Q_21 = grid[y_2, x_2] # bottom right corner
    
    x_1 = tf.cast(x_1, tf.float32)
    y_1 = tf.cast(y_1, tf.float32)
    x_2 = tf.cast(x_2, tf.float32)
    y_2 = tf.cast(y_2, tf.float32)
    
    R_1 = ((x_2 - x)/(x_2 - x_1)) * Q_11 + ((x - x_1)/(x_2 - x_1)) * Q_21 # bottom x interpolation for fixed y_1 (f(x, y_1))
    R_2 = ((x_2 - x)/(x_2 - x_1)) * Q_12 + ((x - x_1)/(x_2 - x_1)) * Q_22 # top x interpolation for fixed y_2 (f(x, y_2))
    
    P = ((y_2 - y)/(y_2 - y_1)) * R_1 + ((y - y_1)/(y_2 - y_1)) * R_2 # final interpolation for fixed x (f(x, y))
    
    return P

tf.function
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

@cuda.jit
def increment_a_2D_array(an_array):
    x, y = cuda.grid(2)
    if x < an_array.shape[0] and y < an_array.shape[1]:
       an_array[x, y] += 1

def main():
    # # Create a 2D grid
    # grid = tf.constant([[91, 210],
    #                    [162, 95]], dtype=tf.float32)
    
    # # Define query points
    # query_points = tf.constant([0.5, 0.2], dtype=tf.float32) # x, y coordinates
    
    # # Interpolate the grid at the query points
    # interpolated_values = interpolate_2d(grid, x=query_points[0], y=query_points[1])

    # assert interpolated_values.numpy() == 146.1
    
    # # Much larger array example
    # Nz = 10
    # Nx = Ny = 10
    # grid = tf.random.uniform((Nz, Nx, Ny, 1), minval=0, maxval=255, dtype=tf.float32)
    
    # # More query points
    # query_points = tf.random.uniform((1, 1000, 2), minval=0, maxval=Nx - 1, dtype=tf.float32)  # 1000 random (x, y) points
    
    # # Interpolate the grid at the query points
    # result_tf = interpolate_bilinear_tf(
    #     grid,
    #     query_points,
    #     indexing="xy",
    # )
    # grid_cp = cp.asarray(grid.numpy())
    # # coordinates_cp_z = cp.linspace(0, grid.shape[0] - 1, grid.shape[0])
    # coordinates_cp_x = cp.linspace(0, grid.shape[1] - 1, grid.shape[1])
    # coordinates_cp_y = cp.linspace(0, grid.shape[2] - 1, grid.shape[2])
    # points = (
    #     # coordinates_cp_z,
    #     coordinates_cp_x,
    #     coordinates_cp_y,
    # )
    # print(grid_cp.shape)
    # query_points_cp = cp.asarray(query_points.numpy())
    # result_cupy = interpn(
    #     points=points,
    #     values=grid_cp,
    #     xi=query_points_cp,
    #     method="cubic",
    # )
    # print(result_tf)
    # print(result_cupy)
    
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(an_array.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(an_array.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    increment_a_2D_array[blockspergrid, threadsperblock](an_array)
    
if __name__ == "__main__":
    main()