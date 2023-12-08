### <h1 align="center" id="title">IGM module `anim_3d_from_ncdf_ex` </h1>

# Description:

This IGM modules makes a 3D animated plot using from the NetCDF output  (default output.nc) file produced by module `write_ncdf`. It only implements the 'finalize' function, the rest ('initialize', 'update') doing nothing.

This module depends on the `mayavi` and `pyqt5` libraryies, which are not included in the original igm package. Therefore, you need to install it in your python environent:

```bash
pip install mayavi pyqt5
```

**Warning: It seems that this module work only with Python <= 3.10**
 
# Parameters: 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--working_dir`|``|Working directory (default empty string)|
||`--modules_preproc`|`['oggm_shop']`|List of pre-processing modules|
||`--modules_process`|`['iceflow', 'time', 'thk']`|List of processing modules|
||`--modules_postproc`|`['write_ncdf', 'plot2d', 'print_info']`|List of post-processing modules|
||`--logging`||Activate the looging|
||`--logging_file`|``|Logging file name, if empty it prints in the screen|
||`--print_params`||Print definitive parameters in a file for record|
