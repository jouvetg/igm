#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf


def initialize(cfg, state):
    state.tlast_mb = tf.Variable(-1.0e5000)


# Warning: The decorator permits to take full benefit from efficient TensorFlow operation (especially on GPU)
# Note that tf.function works best with TensorFlow ops; NumPy and Python calls are converted to constants.
# Therefore: you must make sure any variables are TensorFlow Tensor (and not Numpy)
# @tf.function()
def update(cfg, state):
    """
    mass balance forced by climate with accumulation and temperature-index melt model
    Input:  state.precipitation [Unit: kg * m^(-2) * y^(-1) water eq]
            state.air_temp      [Unit: °C           ]
    Output  state.smb           [Unit: m ice eq. / y]

    This mass balance routine implements a combined accumulation / temperature-index model [Hock, 2003].
    It is a TensorFlow re-implementation similar to the one used in the aletsch-1880-2100 example
    but adapted to fit as closely as possible (thought it is not a strict fit)
    the Positive Degree Day model implemented in PyPDD (Seguinot, 2019) used for the Parralel Ice Sheet
    Model (PISM, Khroulev and the PISM Authors, 2020) necessary to perform PISM / IGM comparison.
    The computation of the PDD using the expectation integration formulation (Calov and Greve, 2005),
    the computation of the snowpack, and the refereezing parameters are taken from PyPDD / PISM implementation.

    References:

    Hock R. (2003). Temperature index melt modelling in mountain areas, J. Hydrol.

    Seguinot J. (2019). PyPDD: a positive degree day model for glacier surface mass balance (v0.3.1).
    Zenodo. https://doi.org/10.5281/zenodo.3467639

    Khroulev C. and the PISM Authors. PISM, a Parallel Ice Sheet Model v1.2: User’s Manual. 2020.
    www.pism-docs.org

    Calov and Greve (2005), A semi-analytical solution for the positive degree-day model with
    stochastic temperature variations, JOG.
    """

    # update smb each X years
    if (state.t - state.tlast_mb) >= cfg.processes.smb.accpdd.update_freq:
        if hasattr(state, "logger"):
            state.logger.info(
                "Construct mass balance at time : " + str(state.t.numpy())
            )

        # keep solid precipitation when temperature < smb_accpdd_thr_temp_snow
        # with linear transition to 0 between smb_accpdd_thr_temp_snow and smb_accpdd_thr_temp_rain
        accumulation = tf.where(
            state.air_temp <= cfg.processes.smb.accpdd.thr_temp_snow,
            state.precipitation,
            tf.where(
                state.air_temp >= cfg.processes.smb.accpdd.thr_temp_rain,
                0.0,
                state.precipitation
                * (cfg.processes.smb.accpdd.thr_temp_rain - state.air_temp)
                / (
                    cfg.processes.smb.accpdd.thr_temp_rain
                    - cfg.processes.smb.accpdd.thr_temp_snow
                ),
            ),
        )

        if hasattr(state, "air_temp_sd"):
            # compute the positive temp with the integral formaulation from Calov and Greve (2005)
            # the formulation assumes the air temperature follows a normal distribution, it is obtained
            # by integrating by part of the integral of T * normal density over {T>=0} since
            # only positive temp. matters. This yields to a first boundary terms, and a second
            # involving the integral of the normal density, i.e. the erf function.
            cela = state.air_temp / (1.4142135623730951 * state.air_temp_sd)
            pos_temp_year = (
                state.air_temp_sd
                * tf.math.exp(-tf.math.square(cela))
                / 2.5066282746310002
                + state.air_temp * tf.math.erfc(-cela) / 2.0
            )
        else:
            pos_temp_year = tf.where(state.air_temp > 0.0, state.air_temp, 0.0)

        # unit to [  kg * m^(-2) * y^(-1) water eq  ] -> [ m water eq ]
        accumulation /= accumulation.shape[0] * cfg.processes.smb.accpdd.wat_density

        # unit to [ °C ]  -> [ °C y ]
        pos_temp_year /= pos_temp_year.shape[0]

        # snowpack / ablation recursion over the (hydrological) year, compiled with
        # tf.function: the former Python loop of eager ops over the time steps was
        # CPU-bound (x4 slower with daily, 365-step climate). Same recursion; the yearly
        # sum differs from the loop by float round-off only.
        ablation = _annual_ablation(
            accumulation,
            pos_temp_year,
            tf.constant(
                int(state.air_temp.shape[0] * cfg.processes.smb.accpdd.shift_hydro_year)
            ),
            tf.constant(cfg.processes.smb.accpdd.melt_factor_ice, dtype=tf.float32),
            tf.constant(cfg.processes.smb.accpdd.melt_factor_snow, dtype=tf.float32),
        )  # [ unit : water-eq m ], one slice per time step

        ablation = (1 - cfg.processes.smb.accpdd.refreeze_factor) * ablation

        # sum accumulation and ablation over the year, and conversion to ice equivalent
        state.smb = (
            tf.math.reduce_sum(accumulation, axis=0)
            - tf.math.reduce_sum(ablation, axis=0)
        ) * (
            cfg.processes.smb.accpdd.wat_density / cfg.processes.smb.accpdd.ice_density
        )

        if hasattr(state, "icemask"):
            state.smb = tf.where(
                (state.smb < 0) | (state.icemask > 0.5),
                state.smb,
                cfg.processes.smb.smb_outside_icemask,
            )

        state.smb = tf.clip_by_value(
            state.smb, -100, cfg.processes.smb.accpdd.smb_maximum_accumulation
        )

        state.tlast_mb.assign(state.t)


@tf.function
def _annual_ablation(accumulation, pos_temp_year, shift, melt_factor_ice, melt_factor_snow):
    """Ablation at each time step of the year [m water eq.], steps visited from `shift`
    (hydrological year): the snow depth is carried from step to step; ice melt factor once
    the snow is gone, snow melt factor while it lasts, a mix of both in the step it vanishes."""
    n = tf.shape(accumulation)[0]
    idx = (tf.range(n) + shift) % n
    acc = tf.gather(accumulation, idx)
    pos = tf.gather(pos_temp_year, idx)

    def step(carry, x):
        snow_depth, _ = carry
        a, p = x
        # add accumulation to the snow depth
        snow_depth = snow_depth + a
        ablation = tf.where(
            snow_depth == 0,
            p * melt_factor_ice,
            tf.where(
                p * melt_factor_snow < snow_depth,
                p * melt_factor_snow,
                snow_depth + (p - snow_depth / melt_factor_snow) * melt_factor_ice,
            ),
        )
        # remove snow melt to snow depth, and cap it as snow_depth can not be negative
        snow_depth = tf.clip_by_value(snow_depth - ablation, 0.0, 1.0e10)
        return snow_depth, ablation

    zero = tf.zeros_like(acc[0])
    _, ablation = tf.scan(step, (acc, pos), initializer=(zero, zero))
    return ablation


def finalize(cfg, state):
    pass
