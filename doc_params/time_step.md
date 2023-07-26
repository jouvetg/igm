
### <h1 align="center" id="title">IGM module time_step </h1>

# Description:

This IGM modules compute time step dt (computed to satisfy the CFL condition),
updated time t, and a boolean telling whether results must be saved or not.
For stability reasons of the transport scheme for the ice thickness evolution,
the time step must respect a CFL condition, controlled by parameter params.cfl,
which is the maximum number of cells crossed in one iteration
(this parameter cannot exceed one).

# I/O

Input  : state.ubar, state.vbar, state.dx 
Output : state.dt, state.t, state.it, state.saveresult 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--tstart`|`2000.0`|Start modelling time (default 2000)|
||`--tend`|`2100.0`|End modelling time (default: 2100)|
||`--tsave`|`10`|Save result each X years (default: 10)|
||`--cfl`|`0.3`|CFL number for the stability of the mass conservation scheme,         it must be below 1 (Default: 0.3)|
||`--dtmax`|`10.0`|Maximum time step allowed, used only with slow ice (default: 10.0)|
 
# Parameters: 
