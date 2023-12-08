### <h1 align="center" id="title">IGM module `anim_3d_from_ncdf_ex` </h1>

# Description:

This IGM modules makes a 3D animated plot using from the NetCDF output  (default output.nc) file produced by module `write_ncdf`. It only implements the 'finalize' function, the rest ('initialize', 'update') doing nothing.

This module depends on the `mayavi` and `pyqt5` libraryies, which are not included in the original igm package. Therefore, you need to install it in your python environent:

```bash
pip install mayavi pyqt5
```

**Warning: It seems that this module work only with Python <= 3.10**
