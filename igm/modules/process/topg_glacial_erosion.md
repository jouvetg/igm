
### <h1 align="center" id="title">IGM module topg_glacial_erosion </h1>

# Description:

This IGM module implements change in basal topography (due to glacial erosion
The bedrock is updated (each params.erosion_update_freq years) assuming the erosion
rate to be proportional (parameter params.erosion_cst) to a power (parameter params.erosion_exp)
of the sliding velocity magnitude. By default, we use the parameters from Herman,
F. et al. Erosion by an Alpine glacier. Science 350, 193-195 (2015).

The module takes as input (state.ubar, state.vbar, state.dx) and return as output 
(state.dt, state.t, state.it, state.saveresult)
