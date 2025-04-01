# import tensorflow as tf
from numba import cuda
import nvtx
import math
import cupy as cp

# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
def srange(message, color):
    tf.test.experimental.sync_devices()
    return nvtx.start_range(message, color)

def erange(rng):
    tf.test.experimental.sync_devices()
    nvtx.end_range(rng)

@cuda.jit
def increment_a_2D_array(an_array):
    x, y = cuda.grid(2)
    if x < an_array.shape[0] and y < an_array.shape[1]:
       an_array[x, y] += 1

@cuda.jit
def interpolate_2d(interpolated_particle_array, array, array_particles):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    # y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    # x, y = cuda.grid(2)
    if x < array_particles.shape[0]:
        x_pos = array_particles[x, 0]
        y_pos = array_particles[x, 1]
        
        x_1 = int(x_pos) # left x coordinate
        y_1 = int(y_pos) # bottom y coordinate
        x_2 = x_1 + 1 # right x coordinate
        y_2 = y_1 + 1 # top y coordinate
        
        Q_11 = array[y_1, x_1] # bottom left corner
        Q_12 = array[y_2, x_1] # top left corner
        Q_21 = array[y_1, x_2] # bottom right corner
        Q_22 = array[y_2, x_2] # top right corner
        
        # Q_12 = array[x_1, y_2] # top left corner
        # Q_22 = array[x_2, y_2] # top right corner
        # Q_11 = array[x_1, y_1] # bottom left corner
        # Q_21 = array[x_2, y_1] # bottom right corner
        
        dx = (x_2 - x_1)
        x_left_weight = (x_pos - x_1) / dx
        x_right_weight = (x_2 - x_pos) / dx
        R_1 = x_left_weight * Q_11 + x_right_weight * Q_21 # bottom x interpolation for fixed y_1 (f(x, y_1))
        R_2 = x_left_weight * Q_12 + x_right_weight * Q_22 # top x interpolation for fixed y_2 (f(x, y_2))
        
        dy = (y_2 - y_1)
        y_bottom_weight = (y_pos - y_1) / dy
        y_top_weight = (y_2 - y_pos) / dy
        
        P = y_bottom_weight * R_1 + y_top_weight * R_2 # final interpolation for fixed x (f(x, y))
        
        interpolated_particle_array[x] = P
        
        # array[x, 0] = dx
        # array[x, 1] = R_2

def main():
    # array_input = cuda.device_array((8192, 8192), dtype="float32")
    
    # threadsperblock = (32, 32)
    # blockspergrid_x = math.ceil(array_input.shape[0] / threadsperblock[0])
    # blockspergrid_y = math.ceil(array_input.shape[1] / threadsperblock[1])
    # blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    # with nvtx.annotate("increment_a_2D_array", color="red"):
    #     increment_a_2D_array[blockspergrid, threadsperblock](array_input)
    res = 4096
    # number_of_particles = 5_000_000
    number_of_particles = 20_000_000
    particles = cp.random.uniform(0, res - 1, size=(number_of_particles, 2))
    grid_x = cp.linspace(0, res, num=res)
    grid_y = cp.linspace(0, res, num=res)
    grid = cp.add(*cp.meshgrid(grid_x, grid_y))
    
    grid = cuda.to_device(grid)
    
    # array_interpolate = cuda.device_array((32, 32), dtype="float32")
    array_particles = cuda.to_device(particles)
    
    interpolated_particle_array = cuda.device_array(shape=particles.shape[0], dtype="float32")
    
    threadsperblock = 32
    blockspergrid = math.ceil(number_of_particles / threadsperblock)
    print("Blocks per grid", blockspergrid)
    # with nvtx.annotate("interpolate_2d", color="blue"): # need to sync GPU first
    rng = srange("interpolate_2d", color="blue")
    interpolate_2d[blockspergrid, threadsperblock](interpolated_particle_array, grid, array_particles)
    erange(rng)
    
    particles_x = cp.asnumpy(particles[:, 0])
    particles_y = cp.asnumpy(particles[:, 1])
    interpolated_values = interpolated_particle_array.copy_to_host()
    
    # print(array_interpolate.copy_to_host())
    plt.imshow(grid.copy_to_host(), cmap="hot_r", origin="lower")
    
    plt.colorbar()
    plt.scatter(particles_x, particles_y, color="blue", s=0.2)
    
    skip_n = 100000
    shortened_particle_list = interpolated_values[::skip_n] # every nth one
    shorted_particles_x = particles_x[::skip_n]
    shorted_particles_y = particles_y[::skip_n]
    
    for i in range(shortened_particle_list.shape[0]):
        print(str(shortened_particle_list[i]))
        plt.text(int(shorted_particles_x[i]), int(shorted_particles_y[i]), str(shortened_particle_list[i]), color="black", fontsize=8)
    
    plt.show()
    # print("Particle Locations", particles)
    # print("Grid Values", grid.copy_to_host())
    # print("Interpolated Values", interpolated_particle_array.copy_to_host())
    
if __name__ == "__main__":
    main()