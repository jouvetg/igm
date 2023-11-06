### <h1 align="center" id="title">IGM module `write_tif` </h1>

# Description:

This IGM module writes 2D field variables defined in the paramer list `wtif_vars_to_save` into tif output files. Files will be created with names composed by the variable name and the time (e.g., thk-000040.tif, usurf-000090.tif) in the working directory. The saving frequency is given by parameter `time_save` defined in module `time`. If input file were call with module `load_tif`, then the tif meta information are saved, and provided with the final tiff files.

This module depends on the `rasterio` library.



 