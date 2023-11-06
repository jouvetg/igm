### <h1 align="center" id="title">IGM avalanche module  </h1>

# Description:

This IGM module permits to model redistribution of snow due to avalanches.
This routine move ice/snow downslope until the ice surface is everywhere
at angle of repose. This function was adapted from 
[Mark Kessler's GC2D](https://github.com/csdms-contrib/gc2d)
program and implemented in IGM by Jürgen Mey with support from Guillaume Jouvet.
 
 
# Parameters: 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--avalanche_update_freq`|`1`|Update avalanche each X years (1)|
||`--avalanche_angleOfRepose`|`30`|Angle of repose (30°)|
