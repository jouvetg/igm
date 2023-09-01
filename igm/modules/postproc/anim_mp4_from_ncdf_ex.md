### <h1 align="center" id="title">IGM module `anim_mp4_from_ncdf_ex`` </h1>

# Description:

This IGM module makes an animated mp4 video of ice thickness over time from 
the NetCDF output (default output.nc) file produced by module `write_ncdf_ex`. It only implements the 'finalize' function, the rest ('initialize', 'update') doing nothing.

This module depends on `xarray` library.
