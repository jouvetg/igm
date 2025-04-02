import tensorflow as tf
import nvtx

def srange(message, color):
    tf.test.experimental.sync_devices()
    return nvtx.start_range(message, color)

def erange(rng):
    tf.test.experimental.sync_devices()
    nvtx.end_range(rng)
    
def gather():
    pass

def main():
    
    num_indices = 2500
    num_grid = 200_000
    grid = tf.random.uniform(shape=(num_grid,1))
    indices = tf.cast(tf.random.uniform(shape=(num_indices, 1), minval=0, maxval=num_indices - 1), tf.int32)
    # indices = tf.cast(tf.random.uniform(shape=(10, num_indices), minval=0, maxval=num_indices - 1), tf.int32)
    
    rng = srange("tensorflow gather", color="red")
    result_gather = tf.gather(grid, indices)
    erange(rng)
    
    rng = srange("tensorflow get items", color="blue")
    for i in range(num_indices):
        _ = grid[indices[i][0]]
    erange(rng)
    
if __name__ == "__main__":
    main()