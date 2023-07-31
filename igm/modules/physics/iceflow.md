### <h1 align="center" id="title">IGM module iceflow </h1>

# Description:

This IGM module models ice flow using a Convolutional Neural Network based on 
Physics Informed Neural Network as described in this 
[paper](https://eartharxiv.org/repository/view/5335/).
You may find pre-trained and ready-to-use ice 
flow emulators, e.g. using the default emulator = f21_pinnbp_GJ_23_a, or using 
an initial untrained with emulator =''. Most important parameters are

- physical parameters (init_slidingco,init_arrhenius, exp_glen,exp_weertman)
- vertical pdiscrezation params (Nz,vert_spacing)
- learning rate (retrain_iceflow_emulator_lr)
- retraining frequency (retrain_iceflow_emulator_freq)
- CNN parameters (...)

While this module was targeted for deep learning emulation, it is possible to
use the solver (setting params.type_iceflow='solved'), or use the two together
(setting params.type_iceflow='diagnostic') to assess the emaultor against the
solver. Most important parameters for solving are :

- solve_iceflow_step_size
- solve_iceflow_nbitmax


