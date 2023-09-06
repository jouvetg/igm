### <h1 align="center" id="title">IGM module `include_icemask` </h1>

# Description:

This IGM module loads a shapefile (ESRI shapefile) and creates an ice mask from it.
The shapefile can be either the coordinates where there should be no glacier (default)
or where there should be glaciers ('mask_invert' = True). 

Input: Shapefile (.shp) exported from any GIS program (e.g. QGIS).
Output: state.icemask

This module can be used with any igm setup that calculates the new glacier surface via the state.smb variable.
    Add to 'smb_simple.py':
        # if an icemask exists, then force negative smb to the side to prevent leakage
        if hasattr(state, "icemask")
            state.smb = tf.where((state.smb<0)|(state.icemask>0.5),state.smb,-10)

The input can be one or more polygon features. Sometimes it is easier to select the valley where the glacier should be ('mask_invert' = True)
or draw polygons where the glacier should not be (e.g. side valleys with no further interest).

IMPORTANT: Be aware of the coordinate system used in the nc file and the shapefile.
 
# Parameters: 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--mask_shapefile`|`icemask.sho`|Icemask input shapefile|
||`--mask_invert`|`False`|Invert ice mask if the mask is where the ice should be|

-------
Author: Andreas Henz, andreas.henz@geo.uzh.ch
Date: 12.07.2023