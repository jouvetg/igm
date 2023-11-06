### <h1 align="center" id="title">IGM module `vert_flow` </h1>

# Description:

This IGM module computes the vertical component (providing state.W) of the velocity from the horizontal components (state.U, computed from an emulation of the Blatter-Pattyn model in the module `iceflow`) by integrating the imcompressibility condition. This module is typically needed prior calling module `particle` for 3D particle trajectory integration, or module `enthalpy` for computing 3D advection-diffusion of the enthalpy.

