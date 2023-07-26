
### <h1 align="center" id="title">IGM module thk </h1>

# Description:

This IGM module solves the mass conservation of ice to update the thickness
from ice flow and surface mass balance. The mass conservation equation
is solved using an explicit first-order upwind finite-volume scheme
on a regular 2D grid with constant cell spacing in any direction.
The discretization and the approximation of the flux divergence is
described [here](https://github.com/jouvetg/igm/blob/main/fig/transp-igm.jpg).
With this scheme mass of ice is allowed to move from cell to cell
(where thickness and velocities are defined) from edge-defined fluxes
(inferred from depth-averaged velocities, and ice thickness in upwind direction).
The resulting scheme is mass conservative and parallelizable (because fully explicit).
However, it is subject to a CFL condition. This means that the time step
(defined in update_t_dt()) is controlled by parameter params.cfl,
which is the maximum number of cells crossed in one iteration
(this parameter cannot exceed one).

# I/O

Input  : state.ubar, state.vbar, state.thk, state.dx, 
Output : state.thk, state.usurf, state.lsurf
 
# Parameters: 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
