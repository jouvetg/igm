### <h1 align="center" id="title">IGM clim_oggm module  </h1>

# Description:

Module `clim_oggm` reads monthly time series of historical GSWP3_W5E5 climate data collected by the `oggm_shop` module, and generates monthly 2D raster fields of corrected precipitation, mean temperature, and temperature variability. To achieve this, we first apply a multiplicative correction factor for precipitation (parameter `prcp_fac`) and a biais correction for temperature (parameter `temp_bias`). Then, the module extrapolates temperature data to the entire glacier surface using a reference height and a constant lapse rate (parameter `temp_default_gradient`). In constrast, the point-wise data for precipitation and temperature variablity are extended to the entire domain without further correction. Module `oggm_shop` provides all calibrated parameters. The resulting fields are intended to be used to force the surface mass balance or enthalpy models.

In addition, this module can generate climate outside the time frame of available data. To that aim, we define a reference period with parameter `clim_oggm_ref_period` to pick randomly years within this interval (usually taken to be a climate-neutral period), and apply a biais in temperature and a scaling of precipitation. These parameters may be given in file (file name given in `clim_oggm_file` parameter), which look like this (this example gives an linear increase of temperature of 4 degrees by the end of 2100 (with respect to the period 1960-1990):

```dat
time   delta_temp  prec_scal
1900          0.0        1.0
2020          0.0        1.0
2100          4.0        1.0
```

 or directly as parameter in the config `params.json` file:

```json
"clim_oggm_clim_trend_array": [ 
                     ["time", "delta_temp", "prec_scal"],
                     [ 1900,           0.0,         1.0],
                     [ 2020,           0.0,         1.0],
                     [ 2100,           4.0,         1.0]
                              ],  
```

If parameter `clim_oggm_clim_trend_array` is set to empty list `[]`, then it will read the file `clim_oggm_file`, otherwise it read the array `clim_oggm_clim_trend_array` (which is here in fact a list of list).
