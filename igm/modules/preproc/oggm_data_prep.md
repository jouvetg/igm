### <h1 align="center" id="title">IGM module prepare_data </h1>

# Description:

This IGM module uses OGGM utilities and GlaThiDa dataset to prepare data 
for the IGM model for a specific glacier given the RGI ID (parameter `RGI_ID`), check at [GLIMS VIeWER](https://www.glims.org/maps/glims) to find the RGI ID of your glacier. By default, data are already posprocessed (parameter `preprocess` is True), with spatial resolution of 100 and a border size of 30. For custom spatial resolution and size of 'border' to keep a safe distance to the glacier margin, one need to set `preprocess` parameter to False, and set `dx` and `border` parameter as desired. 

The module directly provide IGM with all 2D gridded variables (as tensorflow object), and are accessible in the code with e.g. `state.thk`. By default a copy of all the data are stored in a NetCDF file `input_saved.nc` so that this file can be readed directly in a second run with module `load_ncdf_data` instead of re-downloading the data with `oggm_data_prep` again. The module provides all data variables necessary to run IGM for a forward glacier evolution run (assuming we provide basal topography `topg` and ice thickness `thk`), or a preliminary data assimilation/ inverse modelling with the `optimize` module further data (typically `icemaskobs`, `thkinit`, `uvelsurf`, `vvelsurf`, `thkobs`, `usurfobs`).

Data are currently based on COPERNIUS DEM 90 for the top surface DEM, the surface ice velocity from (Millan, 2022), the ice thickness from (Millan, 2022) or (farinotti2019) (the user can choose wit parameter `thk_source` between `millan_ice_thickness` or `consensus_ice_thickness` dataset). 

When activating `include_glathida` to True, ice thickness profiles are downloaded from the [GlaThiDa depo](https://gitlab.com/wgms/glathida) and are rasterized with name `thkobs` (pixels without data are set to NaN values.).

The OGGM script was written by Fabien Maussion. The GlaThiDa script was written by Ethan Welty & Guillaume Jouvet.

The module depends (of course) of `oggm` library.
