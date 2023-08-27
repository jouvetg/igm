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
 
# Parameters: 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--RGI_ID`|`RGI60-11.01450`|RGI ID|
||`--preprocess`||Use preprocessing|
||`--dx`|`100`|Spatial resolution (need preprocess false to change it)|
||`--border`|`30`|Safe border margin  (need preprocess false to change it)|
||`--thk_source`|`consensus_ice_thickness`|millan_ice_thickness or consensus_ice_thickness in geology.nc|
||`--include_glathida`||Make observation file (for IGM inverse)|
||`--path_glathida`|`/home/gjouvet/`|Path where the Glathida Folder is store, so that you don't need               to redownload it at any use of the script|
||`--save_input_ncdf`||Write prepared data into a geology file|
