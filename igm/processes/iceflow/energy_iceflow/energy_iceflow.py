import numpy as np 
import tensorflow as tf 
from igm.processes.utils import *

from .utils import gauss_points_and_weigths, get_dz_COND
from ..utils import X_to_fieldin, Y_to_UV

from .cost_gravity_2layers import cost_gravity_2layers
from .cost_shear_2layers import cost_shear_2layers
from .cost_sliding_2layers import cost_sliding_2layers

from .cost_shear import cost_shear
from .cost_sliding import cost_sliding
from .cost_gravity import cost_gravity
from .cost_floating import cost_floating

def iceflow_energy(cfg, U, V, fieldin):
    thk, usurf, arrhenius, slidingco, dX = fieldin

    if cfg.processes.iceflow.Nz <= 2:

        # this was an attempt to allow a variable exponent, but it is not working
        # if Nz == 2:
        #     exp = exp_glen
        # else:
        #     exp = ( (U[:, -1, :, :] - U[:, 0, :, :]) - 2*(U[:, 1, :, :]  - U[:, 0, :, :]) ) \
        #         / ( (U[:,  1, :, :] - U[:, 0, :, :]) -   (U[:, -1, :, :] - U[:, 0, :, :]) + 1.0 )
        #     exp = (exp[:, 1:, 1:] + exp[:, :-1, 1:] + exp[:, 1:, :-1] + exp[:, :-1, :-1]) / 4
        #     exp = tf.clip_by_value(exp,  1, 5)
 
        exp_glen = cfg.processes.iceflow.exp_glen
        exp_weertman = cfg.processes.iceflow.exp_weertman
        regu_glen = cfg.processes.iceflow.regu_glen
        regu_weertman = cfg.processes.iceflow.regu_weertman
        ice_density = cfg.processes.iceflow.ice_density
        gravity_cst = cfg.processes.iceflow.gravity_cst
 
        n, w = gauss_points_and_weigths(ord_gauss=3)
  
  
        Cshear = cost_shear_2layers(thk, arrhenius, U, V, dX, exp_glen, regu_glen, w, n)
        Cslid = cost_sliding_2layers(U, V, slidingco, exp_weertman, regu_weertman)
        Cgrav = cost_gravity_2layers(U, V, thk, usurf, dX, exp_glen, ice_density, gravity_cst, w, n)
        Cfloat = tf.zeros_like(Cshear) # not implemented for 2 layers

        return Cshear, Cslid, Cgrav, Cfloat
 
    else:

        Nz = cfg.processes.iceflow.Nz
        vert_spacing = cfg.processes.iceflow.vert_spacing
        exp_glen = cfg.processes.iceflow.exp_glen
        exp_weertman = cfg.processes.iceflow.exp_weertman
        regu_glen = cfg.processes.iceflow.regu_glen
        regu_weertman = cfg.processes.iceflow.regu_weertman
        thr_ice_thk = cfg.processes.iceflow.thr_ice_thk
        ice_density = cfg.processes.iceflow.ice_density
        gravity_cst = cfg.processes.iceflow.gravity_cst
        new_friction_param = cfg.processes.iceflow.new_friction_param
        cf_cond = cfg.processes.iceflow.cf_cond
        cf_eswn = cfg.processes.iceflow.cf_eswn
        regu = cfg.processes.iceflow.regu
        min_sr = cfg.processes.iceflow.min_sr
        max_sr = cfg.processes.iceflow.max_sr
        force_negative_gravitational_energy = cfg.processes.iceflow.force_negative_gravitational_energy

        dz, COND = get_dz_COND(thk, Nz, vert_spacing)

        Cshear = cost_shear(U, V, thk, usurf, arrhenius, slidingco, dX, dz, COND, 
                            exp_glen, regu_glen, thr_ice_thk, min_sr, max_sr, regu)

        Cslid =  cost_sliding(U, V, thk, usurf, slidingco, dX, 
                              exp_weertman, regu_weertman, new_friction_param)

        Cgrav = cost_gravity(U, V, usurf, dX, dz, COND, Nz, ice_density, gravity_cst, 
                               force_negative_gravitational_energy)
        
        if cf_cond:
            Cfloat = cost_floating(U, V, thk, usurf, dX, Nz, vert_spacing, cf_eswn)
        else:
            Cfloat = tf.zeros_like(Cshear)

        return Cshear, Cslid, Cgrav, Cfloat
     
def iceflow_energy_XY(cfg, X, Y):
    
    U, V = Y_to_UV(cfg, Y)

    fieldin = X_to_fieldin(cfg, X)

    return iceflow_energy(cfg, U, V, fieldin)
