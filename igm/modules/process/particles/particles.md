
### <h1 align="center" id="title">IGM module `particles` </h1>

# Description:

This IGM module implements a particle tracking routine, which computes trajectory of virtual particles advected by the ice flow. The specificity is that it runs in live time during the forward mdodel run and a large number of particles can be computed tanks to the parrallel implementation with TensorFlow. The routine includes particle seeding (by default in the accumulation area at regular intervals, but this can be customized), and tracking (advection by the velocity field in 3D). There is currently no strategy for removing particles, therefore, there is risk of overloading the memory when using this routine as it is for long time and/or with intense seeding.

 There are currently 2 implementations (switch with parameter `part_tracking_method`:

- `'simple'`: Horizontal and vertical directions are treated differently: i) In the horizontal plan, particles are advected with the horizontal velocity field (interpolated bi-linearly) ii) In the vertical direction, particles are tracked along the ice column scaled between 0 and 1 (0 at the bed, 1 at the top surface) with the  relative position along the ice column. Particles are always initialized at 1 relative height (assumed to be on the surface). The evolution of the particle within the ice column through time is computed according to the surface mass balance: the particle deepens when the surface mass balance is positive (the relative height decreases), and re-emerge when the surface mass balance is negative (the relative height increases).

- `'3d'`: requires to activate module `vert_flow`, which computes the vertical velocity by integrating the divergence of the horizontal velocity. This permits in turn to perform 3D particle tracking.

For now, `part_tracking_method` is by default set to  `'simple'`, as the  `'3d'` method (and the dependence `vert_flow`) needs to further tested.

Note that you my adapt the seeding to your need. You may keep the default seeding in the accumulation area setting the seeding frequency with `part_frequency_seeding` parameter and the seeding density `part_density_seeding` parameter. Alternatively, you may define your own seeding strategy (e.g. seeding close to rock walls/nunataks). To do so, you may redefine the function `seeding_particles()` in a file `particles.py` provided in the working directory (check the example aletsch-1880-2100). When excuted, `igm_run` will overide the original function `seeding_particles()` with the new user-defined one.

The module needs horizontal velocities (state.U), as well as vertical speeds (state.W) that ice computed with the vert_flow module when `part_tracking_method` is set to `3d`. 

**Note:** in the code, positions of particles are recorded within a vector of lenght te number of traked particels state.xpos, state.ypos, state.zpos. Variable state.rhpos provide the relative height within the ice column (1 at the surface, 0 at the bed). At each time step, the weight of surface debris contains in each cell the 2D
 horizontal grid is computed, and stored in variable state.weight_particles.