### <h1 align="center" id="title">IGM module oggm_shop </h1>

# Description:

This IGM module uses OGGM utilities and GlaThiDa dataset to prepare data 
for the IGM model for a specific glacier given the RGI ID (parameter `oggm_RGI_ID`), check at [GLIMS VIeWER](https://www.glims.org/maps/glims) to find the RGI ID of your glacier (only for RGI 6.0 - if using RGI 7.0, please download the RGI shapefiles and use the enquire function in your GIS software to find the right ID - by default, IGM looks at RGI 7.0C [here](https://nsidc.org/data/nsidc-0770/versions/7). but modifying line 304 of the module to '70G' instead of '70C' will give you that version instead. This may be made a parameter in the future once the RGI 7.0 release and associated OGGM data are all finalised). By default, data are already processed (parameter `oggm_preprocess` is True), with spatial resolution of 100 m and an oggm_border size of 30 m. For custom spatial resolution and size of 'oggm_border' to keep a safe distance to the glacier margin, one need to set `oggm_preprocess` parameter to False, and set `oggm_dx` and `oggm_border` parameter as desired. 

The module directly provides IGM with all 2D gridded variables (as tensorflow object), and are accessible in the code with e.g. `state.thk`. By default a copy of all the data are stored in a NetCDF file `input_saved.nc` so that this file can be readed directly in a second run with module `load_ncdf` instead of re-downloading the data with `oggm_shop` again. The module provides all data variables necessary to run IGM for a forward glacier evolution run (assuming we provide basal topography `topg` and ice thickness `thk`), or a preliminary data assimilation/ inverse modelling with the `optimize` module further data (typically `icemaskobs`, `thkinit`, `uvelsurf`, `vvelsurf`, `thkobs`, `usurfobs`).

Data are currently based on COPERNICUS DEM 90 for the top surface DEM, the surface ice velocity from (Millan, 2022), the ice thickness from (Millan, 2022) or (farinotti2019) (the user can choose with parameter `oggm_thk_source` between `millan_ice_thickness` or `consensus_ice_thickness` dataset). 

When activating `oggm_include_glathida` to True, ice thickness profiles are downloaded from the [GlaThiDa depo](https://gitlab.com/wgms/glathida) and are rasterized with name `thkobs` (pixels without data are set to NaN values.) if using RGI 6.0. With RGI 7.0, the GlaThiDa data are downloaded for the specific glacier (defined by the RGI ID) from the OGGM server and are found as a text file in the download folder created by this module (glathida_data.csv), from where they are subsequently read in, rasterised, and NaNs added where there are no observations.

The OGGM script was written by Fabien Maussion. The GlaThiDa script was written by Ethan Welty & Guillaume Jouvet. RGI 7.0 modifications were written by Samuel Cook.

The module depends (of course) on the `oggm` library. Unfortunately the module works on linux and Max, but not on windows (unless using WSL).
