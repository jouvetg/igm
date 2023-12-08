### <h1 align="center" id="title">IGM module load_tif </h1>

# Description:

This IGM module loads spatial 2D raster data from any tif file present in the working directory folder, and transform each of them into tensorflow variables, the name of the file becoming the name of the variable, e.g. the file topg.tif will yield variable topg, ect... It is expected here to import at least basal topography (variable `topg`). It also complete the data, e.g. the basal topography from ice thickness and surface topography. Note that all these variables will therefore be available in the code with `state.myvar` from myvar.tif (e.g. variable `icemask` can be provided, and served to define an accumulation area -- this is usefull for modelling an individual glaciers, and prevent overflowing in neighbouring catchements). The module also contains two functions for resampling (parameter `ltif_coarsen` should be increased to 2,3,4 ..., default 1 value means no coarsening) and cropping the data (parameter `ltif_crop` should be set to True, and the bounds must be definined as wished).

This module depends on `rasterio`.
