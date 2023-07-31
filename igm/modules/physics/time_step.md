
### <h1 align="center" id="title">IGM module time_step </h1>

# Description:

This IGM modules compute time step dt (computed to satisfy the CFL condition),
updated time t, and a boolean telling whether results must be saved or not.
For stability reasons of the transport scheme for the ice thickness evolution,
the time step must respect a CFL condition, controlled by parameter params.cfl,
which is the maximum number of cells crossed in one iteration
(this parameter cannot exceed one).

The module takes as inputs (state.ubar, state.vbar, state.dx) and return as 
output (state.dt, state.t, state.it, state.saveresult)
