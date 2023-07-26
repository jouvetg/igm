
### <h1 align="center" id="title">IGM module particles </h1>

# Description:

This IGM module implments a particle tracking routine, which can compute 
a large number of trajectories (as it is implemented with TensorFlow to 
run in parallel) in live time during the forward model run. The routine 
produces some seeding of particles (by default in the accumulation area
 at regular intervals), and computes the time trajectory of the resulting 
 particle in time advected by the velocity field in 3D. 
 There are currently 2 implementations:
* 'simple', Horizontal and vertical directions are treated differently: 
i) In the horizontal plan, particles are advected with the horizontal velocity 
field (interpolated bi-linearly). The positions are recorded in vector 
(glacier.xpos,glacier.ypos). ii) In the vertical direction, particles are 
tracked along the ice column scaled between 0 and 1 (0 at the bed, 1 at 
the top surface). The relative position along the ice column is recorded 
in vector glacier.rhpos (same dimension as glacier.xpos and iglaciergm.ypos). 
Particles are always initialized at 1 (assumed to be on the surface). 
The evolution of the particle within the ice column through time is 
computed according to the surface mass balance: the particle deepens when 
the surface mass balance is positive (then igm.rhpos decreases), 
and re-emerge when the surface mass balance is negative.
* '3d', The vertical velocity is reconsructed by integrating the divergence 
of the horizontal velocity, this permits in turn to perform 3D particle tracking. 
state.zpos is the z- position within the ice.
Note that in both case, the velocity in the ice layer is reconstructed from 
bottom and surface one assuming 4rth order polynomial profile (SIA-like)

To include this feature, make sure:
* To adapt the seeding to your need. You may keep the default seeding in the 
accumulation area setting the seeding frequency with igm.config.frequency_seeding 
and the seeding density glacier.config.density_seeding. Alternatively, you may 
define your own seeding strategy (e.g. seeding close to rock walls/nunataks). 
To do so, you may redefine the function seeding_particles.

* At each time step, the weight of surface debris contains in each cell the 2D
 horizontal grid is computed, and stored in variable igm.weight_particles.

# I/O:

Input: U,W
Output: state.xpos, ...
 
# Parameters: 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--tracking_method`|`simple`|Method for tracking particles (3d or simple)|
||`--frequency_seeding`|`50`|Frequency of seeding (default: 10)|
||`--density_seeding`|`0.2`|Density of seeding (default: 0.2)|
