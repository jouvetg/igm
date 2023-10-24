### <h1 align="center" id="title">IGM module `write_ts` </h1>

# Description:

This IGM module writes time serie variables (ice glaciated area and volume) into the NetCDF output file defined by parameter `wts_output_file` (default output_ts.nc). The saving frequency is given by parameter `time_save` defined in module `time`.
 
# Parameters: 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--wts_output_file`|`output_ts.nc`|Output ncdf data file (time serie)|
