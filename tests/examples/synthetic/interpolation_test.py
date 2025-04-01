import tensorflow as tf
# import numba as nb

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
    
    P = ((y_2 - y)/(y_2 - y_1)) * R_2 + ((y - y_1)/(y_2 - y_1)) * R_1 # final interpolation for fixed x (f(x, y))
    
    return P

def main():
    # Create a 2D grid
    grid = tf.constant([[91, 210],
                       [162, 95]], dtype=tf.float32)
    
    # Define query points
    query_points = tf.constant([0.5, 0.2], dtype=tf.float32) # x, y coordinates
    
    # Interpolate the grid at the query points
    interpolated_values = interpolate_2d(grid, x=query_points[0], y=query_points[1])

    assert interpolated_values.numpy() == 146.1
    
    # Much larger array example
    grid = tf.range(0, 10000, dtype=tf.float32)
    grid = tf.reshape(grid, (100, 100))
    
    # More query points
    query_points = tf.random.uniform((1000, 2), minval=0, maxval=99, dtype=tf.float32)  # 1000 random (x, y) points
    
    # Interpolate the grid at the query points
    interpolated_values = []
    for i in range(query_points.shape[0]):
        interpolated_values.append(interpolate_2d(grid, x=query_points[i, 0], y=query_points[i, 1]))
    
    interpolated_values = tf.stack(interpolated_values)
    print(interpolated_values.numpy())
    
if __name__ == "__main__":
    main()