### <h1 align="center" id="title">IGM module `anim_plotly_from_ncdf_ex` </h1>

# Description:

This module visualises the output.nc file produced by module `write_ncdf_ex`. It creates a dash app that can be accessed via a browser. 
The link is printed in the console (usually http://127.0.0.1:8050/). The app shows a 3D plot of the glacier's surface 
on top of the surrounding bedrock. The surface color shows either the ice thickness, the velocity magnitude of the surface 
or the surface mass balance. The shown property can be chosen in the dropdown menu at the top. 
The app also includes a slider at the bottom, which defines the displayed timestep of the glacier simulation. 

This module depends on the `dash` library.

This module was implemented by [Oskar Herrmann](https://github.com/ho11laqe).