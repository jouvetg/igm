
### <h1 align="center" id="title">IGM module `rockflow` </h1>

# Description:

This module extends the ice flow outside the glaciated area, by giving a constant speed and along-slope flow direction. This modules serves to track rock-like particles (with module `particles`) everywhere in ice-free and ice-filled areas, particles being either advected at constant steep (controlled by parameter `rock_flow_speed`) following the stepest gradient of the ice-free terrain in 2D, or by ice flow in 3D.
