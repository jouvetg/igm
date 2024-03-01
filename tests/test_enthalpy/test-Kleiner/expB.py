#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import scipy.io
import tensorflow as tf
import igm
import numpy as np
import matplotlib.pyplot as plt

from igm.modules.process.enthalpy import *
from igm.modules.process.enthalpy.enthalpy import vertically_discretize_tf,TpmpEpmp_from_depth_tf
from igm.modules.process.enthalpy.enthalpy import surf_enthalpy_from_temperature_tf, compute_enthalpy_basalmeltrate, temperature_from_enthalpy_tf

# this file is avilable at https://github.com/WangYuzhe/PoLIM-Polythermal-Land-Ice-Model
# verif = scipy.io.loadmat("sol_analytic/enthB_analy_result.mat")

ttf = 2000.0  # 300000
dt = 1.0

tim = np.arange(0, ttf, dt) + dt  # to put back to 300000

parser = igm.params_core()

params, _ = parser.parse_known_args()

modules_dict = { "modules_preproc": [ ], "modules_process": ["iceflow","enthalpy"], "modules_postproc": [ ] }
        
imported_modules = igm.load_modules(modules_dict)

for module in imported_modules:
    module.params(parser)

params = igm.setup_igm_params(parser, imported_modules)

params.iflo_vert_spacing = 1

PAGlen = 5.3e-24
# in Pa^{-3} s^{-1}
alpha = 4 * np.pi / 180  # inclination angle
ws = -0.2  # # % [m a-1]
Prho = params.iflo_ice_density
Pg = params.iflo_gravity_cst

params.enth_KtdivKc = 10 ** (-5)  # check this value if ok ?
params.enth_till_wat_max = 200
params.enth_drain_rate = 0

thk = tf.Variable(200 * tf.ones((1, 1)))

depth, dz = vertically_discretize_tf(thk, params.iflo_Nz, params.iflo_vert_spacing)

strainheat = tf.Variable(
    2.0
    * PAGlen
    * ((Prho * Pg * np.sin(alpha)) ** 4.0)
    * (depth**4.0)
    * params.enth_spy
)
strainheat = strainheat / params.enth_spy  # strainheatsec in Pa s^{-1} = W m^{-3}

frictheat = tf.Variable(0.0 * tf.ones((1, 1)))
geoheatflux = tf.Variable(0.0 * tf.ones((1, 1)))  # in W m-2
tillwat = tf.Variable(0.0 * tf.ones((1, 1)))

# Initial enthalpy field
T = tf.Variable((-1.5 + 273.15) * tf.ones((params.iflo_Nz, 1, 1)))
E = tf.Variable(params.enth_ci * (T - 223.15))
omega = tf.Variable(tf.zeros_like(T))
w = tf.Variable(ws * tf.ones_like(T)) / params.enth_spy

surftemp = tf.Variable((-3.0 + 273.15) * tf.ones((1, 1)))

for it, t in enumerate(tim):
    Tpmp, Epmp = TpmpEpmp_from_depth_tf(
        depth,
        params.iflo_gravity_cst,
        params.iflo_ice_density,
        params.enth_claus_clape,
        params.enth_melt_temp,
        params.enth_ci,
        params.enth_ref_temp,
    )

    surfenth = surf_enthalpy_from_temperature_tf(
        surftemp, params.enth_melt_temp, params.enth_ci, params.enth_ref_temp
    )

    E, basalMeltRate = compute_enthalpy_basalmeltrate(
        E,
        Epmp,
        dt * params.enth_spy,
        dz,
        w,
        surfenth,
        geoheatflux,
        strainheat,
        frictheat,
        tillwat,
        params.iflo_thr_ice_thk,
        params.enth_ki,
        params.iflo_ice_density,
        params.enth_water_density,
        params.enth_ci,
        params.enth_ref_temp,
        params.enth_Lh,
        params.enth_spy,
        params.enth_KtdivKc,
        params.enth_drain_ice_column,
    )

    T, omega = temperature_from_enthalpy_tf(
        E, Tpmp, Epmp, params.enth_ci, params.enth_ref_temp, params.enth_Lh
    )

    tillwat = tillwat + dt * (basalMeltRate - params.enth_drain_rate)
    tillwat = tf.clip_by_value(tillwat, 0, params.enth_till_wat_max)

    if it % 100 == 0:
        print(
            "time :",
            t,
            T[0, 0, 0].numpy(),
            tillwat[0, 0].numpy(),
            basalMeltRate[0, 0].numpy(),
        )


hz = np.arange(0, params.iflo_Nz) / (params.iflo_Nz - 1)

fig = plt.figure(figsize=(6, 8))

plt.subplot(131)
plt.plot(E[:, 0, 0] / 1000, hz * 200)
#plt.plot(
#    verif["enthB_analy_E"] / 1000,
#    verif["enthB_analy_z"] * 200,
#    "--r",
#    marker="o",
#    markersize=5,
#    markevery=20,
#)
plt.ylabel("Enthalpy (* 1000)")

plt.subplot(132)
plt.plot(T[:, 0, 0] - 273.15, hz * 200)
#plt.plot(
#    verif["enthB_analy_T"] - 273.15,
#    verif["enthB_analy_z"] * 200,
#    "--r",
#    marker="o",
#    markersize=5,
#    markevery=20,
#)
plt.ylabel("Temperature (Deg Celcius)")

plt.subplot(133)
plt.plot(omega[:, 0, 0] * 100, hz * 200)
#plt.plot(
#    verif["enthB_analy_omega"] * 100,
#    verif["enthB_analy_z"] * 200,
#    "--r",
#    marker="o",
#    markersize=5,
#    markevery=20,
#)
plt.ylabel("Water content (omega)  (/100)")

plt.savefig("KleinerExpB.png", pad_inches=0)

plt.close()
