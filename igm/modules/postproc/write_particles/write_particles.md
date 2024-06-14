### <h1 align="center" id="title">IGM module `write_particles` </h1>

# Description:

This IGM module writes particle time-position in csv files computed by module `particles`. The saving frequency is given by parameter `time_save` defined in module `time`.

The data are stored in folder 'trajectory' (created if does not exist). Files 'traj-TIME.csv' reports the space-time position of the particles at time TIME with the following structure:

```
ID,  state.particle_x,  state.particle_y,  state.particle_z, state.particle_r,  state.particle_t, state.particle_englt, state.particle_topg, state.particle_thk,
X,                  X,                 X,                 X,                X,                 X,                    X,                   X,                  X,
X,                  X,                 X,                 X,                X,                 X,                    X,                   X,                  X,
X,                  X,                 X,                 X,                X,                 X,                    X,                   X,                  X,
```

providing in turn the particle ID, x,y,z positions, the relative height within the ice column, the seeding time, the englacial time, the bedrock altitude and the ice thickness at the position of the particle.

Code originally written by G. Jouvet, improved and tested by Claire-mathilde Stucki
