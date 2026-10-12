#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Flotation: the water level and the grounded mask.

Every flotation test in IGM (thickness, ice flow, error estimator) goes
through :func:`compute_grounded_mask`,

    grounded  <=>  thk + (rho_water / rho_ice) * (topg - water_level) > 0,

and the lower ice surface is ``lsurf = max(topg, water_level - r * thk)``.
``state.water_level`` is therefore always present:

* ``inputs.<method>.water_level.include: true`` gives a uniform sea/lake
  level equal to ``value`` (e.g. ``0.0`` for present-day sea level);
* a ``water_level`` variable in the input file is used as is;
* otherwise the domain has **no ocean** and the field is filled with
  :data:`WATER_LEVEL_NO_OCEAN`.

The "no ocean" level is a finite value far below any bed, so every formula
above degenerates to the land-only case without a special branch: all ice is
grounded, ``lsurf == topg`` exactly, the ocean depth ``max(wl - topg, 0)`` is
zero (no marine calving, full overburden for ``ocean_connected``
hydrology). This is what a synthetic domain with a bed below 0 m and no
ocean (e.g. ISMIP-HOM) needs; a zero default would instead make such ice
float. Mountain glaciers (bed above 0 m) behave identically with either.
"""

from typing import Optional, Sequence

import tensorflow as tf

#: Water level (m) meaning "no ocean"
WATER_LEVEL_NO_OCEAN = -1.0e6


def no_ocean_like(field: tf.Tensor) -> tf.Tensor:
    """Return a "no ocean" water level with the shape and dtype of ``field``."""
    return tf.fill(tf.shape(field), tf.cast(WATER_LEVEL_NO_OCEAN, field.dtype))


def flotation_function(
    thk: tf.Tensor,
    topg: tf.Tensor,
    water_level: tf.Tensor,
    rho_ratio: float,
) -> tf.Tensor:
    """Return the flotation function, positive where ice is grounded.

    ``rho_ratio`` is water density divided by ice density. The function is
    ``thk + rho_ratio * (topg - water_level)`` minus a float32 round-off
    tolerance, so ice within round-off of flotation counts as floating: with
    ``topg = usurf - thk`` (the ice flow), floating ice is exactly at
    flotation.
    """
    dtype = thk.dtype
    topg = tf.cast(topg, dtype)
    phi = thk + tf.cast(rho_ratio, dtype) * (topg - tf.cast(water_level, dtype))
    tol = tf.cast(64.0 * 2.0**-23, dtype) * (thk + tf.abs(topg))  # 64 float32 ulps
    return phi - tol


def compute_grounded_mask(
    thk: tf.Tensor,
    topg: tf.Tensor,
    water_level: tf.Tensor,
    rho_ratio: float,
) -> tf.Tensor:
    """Return a boolean mask where ice is grounded (see ``flotation_function``)."""
    return flotation_function(thk, topg, water_level, rho_ratio) > 0.0


def _all_cell_corners(mask: tf.Tensor) -> tf.Tensor:
    """Return cells for which all four corner nodes are true."""
    return (
        mask[..., :-1, :-1]
        & mask[..., :-1, 1:]
        & mask[..., 1:, :-1]
        & mask[..., 1:, 1:]
    )


def _any_cell_corner(mask: tf.Tensor) -> tf.Tensor:
    """Return cells for which at least one corner node is true."""
    return (
        mask[..., :-1, :-1]
        | mask[..., :-1, 1:]
        | mask[..., 1:, :-1]
        | mask[..., 1:, 1:]
    )


@tf.function(jit_compile=True)
def compute_cell_ice_mask(
    thk: tf.Tensor, grounded: Optional[tf.Tensor] = None
) -> tf.Tensor:
    """Return Q1 cells with mechanically supported velocity degrees of freedom.

    A full four-ice-node cell is valid whether grounded or floating.  A
    partially covered cell is valid only when it contains ice and all four
    corner locations are grounded; this retains land margins without adding
    unsupported degrees of freedom at a floating front.  If ``grounded`` is
    omitted, only full ice cells are returned.
    """
    ice = thk > 0.0
    full_ice = _all_cell_corners(ice)
    if grounded is None:
        return full_ice
    full_grounded = _all_cell_corners(tf.cast(grounded, tf.bool))
    return full_ice | (_any_cell_corner(ice) & full_grounded)


def _pad_cell_mask_to_nodes(
    cell_mask: tf.Tensor,
    top: int,
    bottom: int,
    left: int,
    right: int,
) -> tf.Tensor:
    rank = cell_mask.shape.rank
    if rank is None:
        raise ValueError("cell_mask rank must be statically known.")
    paddings = [[0, 0]] * (rank - 2) + [[top, bottom], [left, right]]
    return tf.pad(cell_mask, paddings)


@tf.function(jit_compile=True)
def compute_node_ice_mask(
    thk: tf.Tensor, grounded: Optional[tf.Tensor] = None
) -> tf.Tensor:
    """Return ice nodes belonging to at least one active Q1 cell."""
    cell_mask = compute_cell_ice_mask(thk, grounded)
    node_support = (
        _pad_cell_mask_to_nodes(cell_mask, 0, 1, 0, 1)
        | _pad_cell_mask_to_nodes(cell_mask, 0, 1, 1, 0)
        | _pad_cell_mask_to_nodes(cell_mask, 1, 0, 0, 1)
        | _pad_cell_mask_to_nodes(cell_mask, 1, 0, 1, 0)
    )
    return (thk > 0.0) & node_support


def iceflow_node_mask(
    thk: tf.Tensor,
    usurf: tf.Tensor,
    water_level: tf.Tensor,
    rho_ratio: float,
) -> tf.Tensor:
    """Ice nodes that carry velocity in the ice flow.

    This is the unified evaluator's node mask: flotation is tested on the
    lower surface ``usurf - thk``, and a node is kept when it belongs to an
    active Q1 cell (:func:`compute_node_ice_mask`). A floating or marine
    front is therefore sharp on the boundary of full cells, while land
    margins keep their partially covered, fully grounded cells.
    """
    grounded = compute_grounded_mask(thk, usurf - thk, water_level, rho_ratio)
    return compute_node_ice_mask(thk, grounded)


def _grounded_length(phi_start: tf.Tensor, phi_end: tf.Tensor) -> tf.Tensor:
    """Return the grounded fraction of a segment on which ``phi`` is linear.

    ``phi_start`` and ``phi_end`` are the flotation function at the two ends.
    The fraction is ``(max(phi_start, 0) + max(phi_end, 0)) / (|phi_start| +
    |phi_end|)``: one when both ends are grounded, zero when both float, and
    the position of the zero crossing otherwise.
    """
    grounded = tf.maximum(phi_start, 0.0) + tf.maximum(phi_end, 0.0)
    total = tf.abs(phi_start) + tf.abs(phi_end)
    nonzero = total > 0.0
    safe_total = tf.where(nonzero, total, tf.ones_like(total))
    return tf.where(nonzero, grounded / safe_total, tf.zeros_like(total))


def _bilinear_grounded_fraction(
    phi_sw: tf.Tensor,
    phi_se: tf.Tensor,
    phi_nw: tf.Tensor,
    phi_ne: tf.Tensor,
    lines: int,
) -> tf.Tensor:
    """Grounded area fraction of rectangles on which ``phi`` is bilinear.

    ``phi_*`` are the corner values. ``phi`` is linear along every
    grid-aligned line, where the grounded length is exact
    (``_grounded_length``). The area is integrated with the midpoint rule over
    ``lines`` such lines in each direction; the two estimates are blended with
    weights proportional to the variation of ``phi`` along each direction,
    which favours the lines crossing the grounding line. The fraction is exact
    when ``phi`` varies along one grid direction only, and continuous in the
    corner values.
    """
    dtype = phi_sw.dtype
    t = (tf.range(lines, dtype=dtype) + 0.5) / tf.cast(lines, dtype)
    t = tf.reshape(t, [lines, 1, 1])

    # Lines along x at eta = t, and along y at xi = t
    sw = phi_sw[..., tf.newaxis, :, :]
    se = phi_se[..., tf.newaxis, :, :]
    nw = phi_nw[..., tf.newaxis, :, :]
    ne = phi_ne[..., tf.newaxis, :, :]
    phi_west = (1.0 - t) * sw + t * nw
    phi_east = (1.0 - t) * se + t * ne
    phi_south = (1.0 - t) * sw + t * se
    phi_north = (1.0 - t) * nw + t * ne
    fraction_x = tf.reduce_mean(_grounded_length(phi_west, phi_east), axis=-3)
    fraction_y = tf.reduce_mean(_grounded_length(phi_south, phi_north), axis=-3)

    # Variation of phi along x and along y
    variation_x = tf.abs(phi_se - phi_sw) + tf.abs(phi_ne - phi_nw)
    variation_y = tf.abs(phi_nw - phi_sw) + tf.abs(phi_ne - phi_se)
    variation = variation_x + variation_y
    nonzero = variation > 0.0
    safe_variation = tf.where(nonzero, variation, tf.ones_like(variation))
    weight_x = tf.where(
        nonzero, variation_x / safe_variation, 0.5 * tf.ones_like(variation)
    )

    return fraction_y + weight_x * (fraction_x - fraction_y)


def weights_gr(
    thk: tf.Tensor,
    topg: tf.Tensor,
    water_level: tf.Tensor,
    rho_ratio: float,
    lines: int = 8,
) -> tf.Tensor:
    """Return the grounded weight of each node, in [0, 1].

    The weight is the grounded fraction of the area around the node: the
    quarters of the (up to four) cells touching it, split from each cell at
    its midlines. The flotation function is bilinear in a cell, hence in each
    quarter, whose corner values are the nodal values, the edge midpoint means
    and the cell-centre mean. It weights the friction inputs with
    ``physics.sliding.use_subgrid_gl`` (the grounding-line parameterization of
    CISM, Leguy et al., 2021). ``lines`` is the number of grid-aligned lines
    per direction of the area integration (``_bilinear_grounded_fraction``).
    Fields have shape ``(..., Ny, Nx)``, as the result.
    """
    phi = flotation_function(thk, topg, water_level, rho_ratio)
    sw = phi[..., :-1, :-1]
    se = phi[..., :-1, 1:]
    nw = phi[..., 1:, :-1]
    ne = phi[..., 1:, 1:]
    south = 0.5 * (sw + se)
    north = 0.5 * (nw + ne)
    west = 0.5 * (sw + nw)
    east = 0.5 * (se + ne)
    centre = 0.25 * (sw + se + nw + ne)

    # Each cell quarter belongs to the corner node it touches
    quarter_sw = _bilinear_grounded_fraction(sw, south, west, centre, lines)
    quarter_se = _bilinear_grounded_fraction(south, se, centre, east, lines)
    quarter_nw = _bilinear_grounded_fraction(west, centre, nw, north, lines)
    quarter_ne = _bilinear_grounded_fraction(centre, east, north, ne, lines)
    grounded = (
        _pad_cell_mask_to_nodes(quarter_sw, 0, 1, 0, 1)
        + _pad_cell_mask_to_nodes(quarter_se, 0, 1, 1, 0)
        + _pad_cell_mask_to_nodes(quarter_nw, 1, 0, 0, 1)
        + _pad_cell_mask_to_nodes(quarter_ne, 1, 0, 1, 0)
    )
    ones = tf.ones_like(quarter_sw)
    quarters = (
        _pad_cell_mask_to_nodes(ones, 0, 1, 0, 1)
        + _pad_cell_mask_to_nodes(ones, 0, 1, 1, 0)
        + _pad_cell_mask_to_nodes(ones, 1, 0, 0, 1)
        + _pad_cell_mask_to_nodes(ones, 1, 0, 1, 0)
    )
    return grounded / quarters


def mask_gr(
    thk: tf.Tensor,
    topg: tf.Tensor,
    water_level: tf.Tensor,
    rho_ratio: float,
) -> tf.Tensor:
    """The grounded mask in ``thk`` dtype (1 grounded, 0 floating)."""
    return tf.cast(compute_grounded_mask(thk, topg, water_level, rho_ratio), thk.dtype)


def water_level_from_inputs(
    inputs: tf.Tensor, input_names: Sequence[str], like: tf.Tensor
) -> tf.Tensor:
    """Return the ``water_level`` channel of ``inputs``, or "no ocean".

    ``water_level`` is not a default iceflow input (it would be a network
    input channel); ``initialize_iceflow_fields`` warns when a domain with an
    ocean does not list it. ``like`` sets shape and dtype.
    """
    if "water_level" in input_names:
        return inputs[..., tuple(input_names).index("water_level")]
    return no_ocean_like(like)
