### <h1 align="center" id="title">IGM module `vert_flow` </h1>

# Description:

This IGM module computes the vertical component (providing state.W) of the velocity from the horizontal components (state.U, computed from an emulation of the Blatter-Pattyn model in the module `iceflow`) by integrating the imcompressibility condition. This module is typically needed prior calling module `particle` for 3D particle trajectory integration, or module `enthalpy` for computing 3D advection-diffusion of the enthalpy.

There are currently two implementations (switch with parameter `vflo_method`):

- `'incompressibility'` is a direct integration of the incompressibility condition $\overrightarrow \nabla .\overrightarrow u = 0$ :
$$ w(z) = - \int_b^z \left( \frac{du}{dx} + \frac{dv}{dy} \right) dz + w(b) $$ 
where $w(b)$ is the basal velocity computed using a condition of bedrock impermeability. This method produces inaccurate results and should be further investigated

- `'kinematic'`is a rewriting of this condition, mathematically and physically equivalent, but which produces results that are physically more meaningsfull.
$$ w(z) = \frac{dz}{dx}u + \frac{dz}{dy}v - \frac{d}{dx}\left( \int_b^z u dz \right) - \frac{d}{dy}\left( \int_b^z v dz \right)$$
where $z(x,y)$ is considered as the surface of constant relative height $r= \frac{z-b}{s-b}$, defined for all value of $r$ between 0 and 1.

`'vflo_method'` is by default set to `'kinematic'`

Code originally written by G. Jouvet, documented and tested by Claire-mathilde Stucki
