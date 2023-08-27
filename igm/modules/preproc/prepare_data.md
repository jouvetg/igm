### <h1 align="center" id="title">IGM module prepare_data </h1>

# Description:

This IGM modules use OGGM utilities and GlaThiDa dataset to prepare data 
for the IGM model for a specific glacier given the RGI ID. One must provide
an RGI ID (check GLIMS VIeWER : https://www.glims.org/maps/glims) 
By default, data are already posprocessed, with spatial resolutio of 100 and 
border of 30. For custom spatial resolution, and the size of 'border' 
to keep a safe distance to the glacier margin, one need
to set preprocess option to False 
The script returns the geology.nc file as necessary for run 
IGM for a forward glacier evolution run, and optionaly 
observation.nc that permit to do a first step of data assimilation & inversion. 
Data are currently based on COPERNIUS DEM 90 
the RGI, and the ice thckness and velocity from (MIllan, 2022) 
For the ice thickness in geology.nc, the use can choose 
between consensus_ice_thickness (farinotti2019) or
millan_ice_thickness (millan2022) dataset 
When activating observation==True, ice thickness profiles are 
downloaded from the GlaThiDa depo (https://gitlab.com/wgms/glathida) 
and are rasterized on working grids 
Script written by G. Jouvet & F. Maussion & E. Welty

The module takes all input variable fields neede to run IGM inverse and/or forward
