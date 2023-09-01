
### <h1 align="center" id="title">IGM module `rockflow` </h1>

# Description:

This module extends the ice flow outside the glaciated area, by giving a constant speed and along-slope flow direction. This modules serves to track rock-like particles (with module `particles`) everywhere in ice-free and ice-filled areas, particles being either advected at constant steep (controlled by parameter `speed_rockflow`) following the stepest gradient of the ice-free terrain in 2D, or by ice flow in 3D.
 
# Parameters: 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--speed_rockflow`|`1`|Speed of rock flow along the slope in m/y|
