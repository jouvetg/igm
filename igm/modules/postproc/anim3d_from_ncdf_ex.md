### <h1 align="center" id="title">IGM module `anim_3d_from_ncdf_ex` </h1>

# Description:

This IGM modules makes a 3D animated plot using from the NetCDF output  (default output.nc) file produced by module `write_ncdf_ex`. It only implements the 'finalize' function, the rest ('initialize', 'update') doing nothing.

This module depends on the `mayavi` library.
