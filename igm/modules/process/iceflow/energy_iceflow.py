import numpy as np 
import tensorflow as tf 
from igm.modules.utils import *


@tf.function(experimental_relax_shapes=True)
def _compute_gradient_stag(s, dX, dY):
    """
    compute spatial gradient, outcome on stagerred grid
    """

    E = 2.0 * (s[:, :, 1:] - s[:, :, :-1]) / (dX[:, :, 1:] + dX[:, :, :-1])
    diffx = 0.5 * (E[:, 1:, :] + E[:, :-1, :])

    EE = 2.0 * (s[:, 1:, :] - s[:, :-1, :]) / (dY[:, 1:, :] + dY[:, :-1, :])
    diffy = 0.5 * (EE[:, :, 1:] + EE[:, :, :-1])

    return diffx, diffy


@tf.function(experimental_relax_shapes=True)
def _compute_strainrate_Glen_tf(U, V, thk, slidingco, dX, ddz, sloptopgx, sloptopgy, thr):
    # Compute horinzontal derivatives
    dUdx = (U[:, :, :, 1:] - U[:, :, :, :-1]) / dX[0, 0, 0]
    dVdx = (V[:, :, :, 1:] - V[:, :, :, :-1]) / dX[0, 0, 0]
    dUdy = (U[:, :, 1:, :] - U[:, :, :-1, :]) / dX[0, 0, 0]
    dVdy = (V[:, :, 1:, :] - V[:, :, :-1, :]) / dX[0, 0, 0]

    # Homgenize sizes in the horizontal plan on the stagerred grid
    dUdx = (dUdx[:, :, :-1, :] + dUdx[:, :, 1:, :]) / 2
    dVdx = (dVdx[:, :, :-1, :] + dVdx[:, :, 1:, :]) / 2
    dUdy = (dUdy[:, :, :, :-1] + dUdy[:, :, :, 1:]) / 2
    dVdy = (dVdy[:, :, :, :-1] + dVdy[:, :, :, 1:]) / 2

    # homgenize sizes in the vertical plan on the stagerred grid
    if U.shape[1] > 1:
        dUdx = (dUdx[:, :-1, :, :] + dUdx[:, 1:, :, :]) / 2
        dVdx = (dVdx[:, :-1, :, :] + dVdx[:, 1:, :, :]) / 2
        dUdy = (dUdy[:, :-1, :, :] + dUdy[:, 1:, :, :]) / 2
        dVdy = (dVdy[:, :-1, :, :] + dVdy[:, 1:, :, :]) / 2

    # compute the horizontal average, these quantitites will be used for vertical derivatives
    Um = (U[:, :, 1:, 1:] + U[:, :, 1:, :-1] + U[:, :, :-1, 1:] + U[:, :, :-1, :-1]) / 4
    Vm = (V[:, :, 1:, 1:] + V[:, :, 1:, :-1] + V[:, :, :-1, 1:] + V[:, :, :-1, :-1]) / 4

    if U.shape[1] > 1:
        # vertical derivative if there is at least two layears
        dUdz = (Um[:, 1:, :, :] - Um[:, :-1, :, :]) / tf.maximum(ddz, thr)
        dVdz = (Vm[:, 1:, :, :] - Vm[:, :-1, :, :]) / tf.maximum(ddz, thr)
        slc = tf.expand_dims(_stag4(slidingco), axis=1)
        dUdz = tf.where(slc > 0, dUdz, 0.01 * dUdz)
        dVdz = tf.where(slc > 0, dVdz, 0.01 * dVdz)
    else:
        # zero otherwise
        dUdz = 0.0
        dVdz = 0.0

    # This correct for the change of coordinate z -> z - b
    dUdx = dUdx - dUdz * sloptopgx
    dUdy = dUdy - dUdz * sloptopgy
    dVdx = dVdx - dVdz * sloptopgx
    dVdy = dVdy - dVdz * sloptopgy

    Exx = dUdx
    Eyy = dVdy
    Ezz = -dUdx - dVdy
    Exy = 0.5 * dVdx + 0.5 * dUdy
    Exz = 0.5 * dUdz
    Eyz = 0.5 * dVdz
    
    srx = 0.5 * ( Exx**2 + Exy**2 + Exy**2 + Eyy**2 + Ezz**2 )
    srz = 0.5 * ( Exz**2 + Eyz**2 + Exz**2 + Eyz**2 )

    return srx, srz


def _stag2(B):
    return (B[:, 1:] + B[:, :-1]) / 2


def _stag4(B):
    return (B[:, 1:, 1:] + B[:, 1:, :-1] + B[:, :-1, 1:] + B[:, :-1, :-1]) / 4


def _stag4b(B):
    return (
        B[:, :, 1:, 1:] + B[:, :, 1:, :-1] + B[:, :, :-1, 1:] + B[:, :, :-1, :-1]
    ) / 4


def _stag8(B):
    return (
        B[:, 1:, 1:, 1:]
        + B[:, 1:, 1:, :-1]
        + B[:, 1:, :-1, 1:]
        + B[:, 1:, :-1, :-1]
        + B[:, :-1, 1:, 1:]
        + B[:, :-1, 1:, :-1]
        + B[:, :-1, :-1, 1:]
        + B[:, :-1, :-1, :-1]
    ) / 8


def iceflow_energy(params, U, V, fieldin):
    thk, usurf, arrhenius, slidingco, dX = fieldin

    return _iceflow_energy(
        U,
        V,
        thk,
        usurf,
        arrhenius,
        slidingco,
        dX,
        params.iflo_Nz,
        params.iflo_vert_spacing,
        params.iflo_exp_glen,
        params.iflo_exp_weertman,
        params.iflo_regu_glen,
        params.iflo_regu_weertman,
        params.iflo_thr_ice_thk,
        params.iflo_ice_density,
        params.iflo_gravity_cst,
        params.iflo_new_friction_param,
        params.iflo_cf_cond,
        params.iflo_cf_eswn,
        params.iflo_regu,
        params.iflo_min_sr,
        params.iflo_max_sr,
        params.iflo_force_negative_gravitational_energy
    )


@tf.function(experimental_relax_shapes=True)
def _iceflow_energy(
    U,
    V,
    thk,
    usurf,
    arrhenius,
    slidingco,
    dX,
    Nz,
    vert_spacing,
    exp_glen,
    exp_weertman,
    regu_glen,
    regu_weertman,
    thr_ice_thk,
    ice_density,
    gravity_cst,
    new_friction_param,
    iflo_cf_cond,
    iflo_cf_eswn,
    iflo_regu,
    min_sr,
    max_sr,
    iflo_force_negative_gravitational_energy
):
    # warning, the energy is here normalized dividing by int_Omega

    COND = (
        (thk[:, 1:, 1:] > 0)
        & (thk[:, 1:, :-1] > 0)
        & (thk[:, :-1, 1:] > 0)
        & (thk[:, :-1, :-1] > 0)
    )
    COND = tf.expand_dims(COND, axis=1)

    # Vertical discretization
    if Nz > 1:
        zeta = np.arange(Nz) / (Nz - 1)  # formerly ...
        #zeta = tf.range(Nz, dtype=tf.float32) / (Nz - 1)
        temp = (zeta / vert_spacing) * (1.0 + (vert_spacing - 1.0) * zeta)
        temd = temp[1:] - temp[:-1]
        dz = tf.stack([_stag4(thk) * z for z in temd], axis=1)  # formerly ..
        #dz = (tf.expand_dims(tf.expand_dims(temd,axis=-1),axis=-1)*tf.expand_dims(_stag4(thk),axis=0))
    else:
        dz = tf.expand_dims(_stag4(thk), axis=0)

    # B has Unit Mpa y^(1/n)
    B = 2.0 * arrhenius ** (-1.0 / exp_glen)

    if new_friction_param:
        C = 1.0 * slidingco  # C has unit Mpa y^m m^(-m)
    else:
        if exp_weertman == 1:
            # C has unit Mpa y^m m^(-m)
            C = 1.0 * slidingco
        else:
            C = (slidingco + 10 ** (-12)) ** -(1.0 / exp_weertman)

    p = 1.0 + 1.0 / exp_glen
    s = 1.0 + 1.0 / exp_weertman

    sloptopgx, sloptopgy = _compute_gradient_stag(usurf - thk, dX, dX)
    sloptopgx = tf.expand_dims(sloptopgx, axis=1)
    sloptopgy = tf.expand_dims(sloptopgy, axis=1)

    # TODO : sloptopgx, sloptopgy must be the elevaion of layers! not the bedrock, this probably has very little effects.

    # sr has unit y^(-1)
    srx, srz = _compute_strainrate_Glen_tf(
        U, V, thk, C, dX, dz, sloptopgx, sloptopgy, thr=thr_ice_thk
    )
    
    sr = srx + srz

    sr = tf.where(COND, sr, 0.0)
    
    srcapped = tf.clip_by_value(sr, min_sr**2, max_sr**2)

    srcapped = tf.where(COND, srcapped, 0.0)
 

    # C_shear is unit  Mpa y^(1/n) y^(-1-1/n) * m = Mpa m/y
    if len(B.shape) == 3:
        C_shear = _stag4(B) * tf.reduce_sum(dz * ((srcapped + regu_glen**2) ** ((p-2) / 2)) * sr, axis=1 ) / p
    else:
        C_shear = tf.reduce_sum( _stag8(B) * dz * ((srcapped + regu_glen**2) ** ((p-2) / 2)) * sr, axis=1 ) / p
        
    if iflo_regu > 0:
        
        srx = tf.where(COND, srx, 0.0)
 
        if len(B.shape) == 3:
            C_shear_2 = _stag4(B) * tf.reduce_sum(dz * ((srx + regu_glen**2) ** (p / 2)), axis=1 ) / p
        else:
            C_shear_2 = tf.reduce_sum( _stag8(B) * dz * ((srx + regu_glen**2) ** (p / 2)), axis=1 ) / p 

        C_shear = C_shear + iflo_regu*C_shear_2
        
    lsurf = usurf - thk

    sloptopgx, sloptopgy = _compute_gradient_stag(lsurf, dX, dX)

    # C_slid is unit Mpa y^m m^(-m) * m^(1+m) * y^(-1-m)  = Mpa  m/y
    N = (
        _stag4(U[:, 0, :, :] ** 2 + V[:, 0, :, :] ** 2)
        + regu_weertman**2
        + (_stag4(U[:, 0, :, :]) * sloptopgx + _stag4(V[:, 0, :, :]) * sloptopgy) ** 2
    )
    C_slid = _stag4(C) * N ** (s / 2) / s

    slopsurfx, slopsurfy = _compute_gradient_stag(usurf, dX, dX)
    slopsurfx = tf.expand_dims(slopsurfx, axis=1)
    slopsurfy = tf.expand_dims(slopsurfy, axis=1)

    if Nz > 1:
        uds = _stag8(U) * slopsurfx + _stag8(V) * slopsurfy
    else:
        uds = _stag4b(U) * slopsurfx + _stag4b(V) * slopsurfy  

    if iflo_force_negative_gravitational_energy:
        uds = tf.minimum(uds, 0.0) # force non-postiveness

    uds = tf.where(COND, uds, 0.0)

    # C_slid is unit Mpa m^-1 m/y m = Mpa m/y
    C_grav = (
        ice_density
        * gravity_cst
        * 10 ** (-6)
        * tf.reduce_sum(dz * uds, axis=1)
    )

    # if activae this applies the stress condition along the calving front
    if iflo_cf_cond:

        ################################################################
        
        lsurf = usurf - thk
        
    #   Check formula (17) in [Jouvet and Graeser 2012], Unit is Mpa 
        P =tf.where(lsurf<0, 0.5 * 10 ** (-6) * 9.81 * 910 * ( thk**2 - (1000/910)*lsurf**2 ) , 0.0)  / dX[:, 0, 0] 
        
        if len(iflo_cf_eswn) == 0:
            thkext = tf.pad(thk,[[0,0],[1,1],[1,1]],"CONSTANT",constant_values=1)
            lsurfext = tf.pad(lsurf,[[0,0],[1,1],[1,1]],"CONSTANT",constant_values=1)
        else:
            thkext = thk
            thkext = tf.pad(thkext,[[0,0],[1,0],[0,0]],"CONSTANT",constant_values=1.0*('S' not in iflo_cf_eswn))
            thkext = tf.pad(thkext,[[0,0],[0,1],[0,0]],"CONSTANT",constant_values=1.0*('N' not in iflo_cf_eswn))
            thkext = tf.pad(thkext,[[0,0],[0,0],[1,0]],"CONSTANT",constant_values=1.0*('W' not in iflo_cf_eswn))
            thkext = tf.pad(thkext,[[0,0],[0,0],[0,1]],"CONSTANT",constant_values=1.0*('E' not in iflo_cf_eswn)) 
            lsurfext = lsurf
            lsurfext = tf.pad(lsurfext,[[0,0],[1,0],[0,0]],"CONSTANT",constant_values=1.0*('S' not in iflo_cf_eswn))
            lsurfext = tf.pad(lsurfext,[[0,0],[0,1],[0,0]],"CONSTANT",constant_values=1.0*('N' not in iflo_cf_eswn))
            lsurfext = tf.pad(lsurfext,[[0,0],[0,0],[1,0]],"CONSTANT",constant_values=1.0*('W' not in iflo_cf_eswn))
            lsurfext = tf.pad(lsurfext,[[0,0],[0,0],[0,1]],"CONSTANT",constant_values=1.0*('E' not in iflo_cf_eswn)) 
        
        # this permits to locate the calving front in a cell in the 4 directions
        CF_W = tf.where((lsurf<0)&(thk>0)&(thkext[:,1:-1,:-2]==0)&(lsurfext[:,1:-1,:-2]<=0),1.0,0.0)
        CF_E = tf.where((lsurf<0)&(thk>0)&(thkext[:,1:-1,2:]==0)&(lsurfext[:,1:-1,2:]<=0),1.0,0.0) 
        CF_S = tf.where((lsurf<0)&(thk>0)&(thkext[:,:-2,1:-1]==0)&(lsurfext[:,:-2,1:-1]<=0),1.0,0.0)
        CF_N = tf.where((lsurf<0)&(thk>0)&(thkext[:,2:,1:-1]==0)&(lsurfext[:,2:,1:-1]<=0),1.0,0.0)
 
        if Nz > 1:
            # Blatter-Pattyn
            weight = tf.stack([tf.ones_like(thk) * z for z in temd], axis=1) # dimensionless, 
            C_float = (
                  P * tf.reduce_sum(weight * _stag2(U), axis=1) * CF_W
                - P * tf.reduce_sum(weight * _stag2(U), axis=1) * CF_E 
                + P * tf.reduce_sum(weight * _stag2(V), axis=1) * CF_S 
                - P * tf.reduce_sum(weight * _stag2(V), axis=1) * CF_N 
            ) 
        else:
            # SSA
            C_float = ( P * U * CF_W - P * U * CF_E  + P * V * CF_S - P * V * CF_N )  

        ###########################################################

        # ddz = tf.stack([thk * z for z in temd], axis=1) 

        # zzz = tf.expand_dims(lsurf, axis=1) + tf.math.cumsum(ddz, axis=1)

        # f = 10 ** (-6) * ( 910 * 9.81 * (tf.expand_dims(usurf, axis=1) - zzz) + 1000 * 9.81 * tf.minimum(0.0, zzz) )  # Mpa m^(-1) 

        # C_float = (
        #       tf.reduce_sum(ddz * f * _stag2(U), axis=1) * CF_W
        #     - tf.reduce_sum(ddz * f * _stag2(U), axis=1) * CF_E 
        #     + tf.reduce_sum(ddz * f * _stag2(V), axis=1) * CF_S 
        #     - tf.reduce_sum(ddz * f * _stag2(V), axis=1) * CF_N 
        # )   # Mpa m / y
        
        ##########################################################

        # f = 10 ** (-6) * ( 910 * 9.81 * thk + 1000 * 9.81 * tf.minimum(0.0, lsurf) ) # Mpa 

 
        # sloptopgx, sloptopgy = compute_gradient_tf(lsurf[0], dX[0, 0, 0], dX[0, 0, 0])
        # slopn = (sloptopgx**2 + sloptopgy**2 + 1.e-10 )**0.5
        # nx = tf.expand_dims(sloptopgx/slopn,0)
        # ny = tf.expand_dims(sloptopgy/slopn,0)
            
        # C_float_2 = - tf.where( (thk>0)&(slidingco==0), - f * (U[:,0] * nx + V[:,0] * ny), 0.0 ) # Mpa m/y

        # #C_float is unit  Mpa m * (m/y) / m + Mpa m / y = Mpa m / y    
        # C_float = C_float + C_float_2 
        
    else:
        C_float = tf.zeros_like(C_shear)

    # print(C_shear[0].numpy(),C_slid[0].numpy(),C_grav[0].numpy(),C_float[0].numpy())

    # C_pen = 10000 * tf.where(thk>0,0.0, tf.reduce_sum( tf.abs(U), axis=1)**2 )

    return C_shear, C_slid, C_grav, C_float


# @tf.function(experimental_relax_shapes=True)
def iceflow_energy_XY(params, X, Y):
    U, V = Y_to_UV(params, Y)

    fieldin = X_to_fieldin(params, X)

    return iceflow_energy(params, U, V, fieldin)


def Y_to_UV(params, Y):
    N = params.iflo_Nz

    U = tf.experimental.numpy.moveaxis(Y[:, :, :, :N], [-1], [1])
    V = tf.experimental.numpy.moveaxis(Y[:, :, :, N:], [-1], [1])

    return U, V


def UV_to_Y(params, U, V):
    UU = tf.experimental.numpy.moveaxis(U, [0], [-1])
    VV = tf.experimental.numpy.moveaxis(V, [0], [-1])
    RR = tf.expand_dims(
        tf.concat(
            [UU, VV],
            axis=-1,
        ),
        axis=0,
    )

    return RR


def fieldin_to_X(params, fieldin):
    X = []

    fieldin_dim = [0, 0, 1 * (params.iflo_dim_arrhenius == 3), 0, 0]

    for f, s in zip(fieldin, fieldin_dim):
        if s == 0:
            X.append(tf.expand_dims(f, axis=-1))
        else:
            X.append(tf.experimental.numpy.moveaxis(f, [0], [-1]))

    return tf.expand_dims(tf.concat(X, axis=-1), axis=0)


def X_to_fieldin(params, X):
    i = 0

    fieldin_dim = [0, 0, 1 * (params.iflo_dim_arrhenius == 3), 0, 0]

    fieldin = []

    for f, s in zip(params.iflo_fieldin, fieldin_dim):
        if s == 0:
            fieldin.append(X[:, :, :, i])
            i += 1
        else:
            fieldin.append(
                tf.experimental.numpy.moveaxis(
                    X[:, :, :, i : i + params.iflo_Nz], [-1], [1]
                )
            )
            i += params.iflo_Nz

    return fieldin
