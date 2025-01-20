import xarray as xr
import numpy as np
import tensorflow as tf

from hydra.utils import get_original_cwd
from pathlib import Path

def run(cfg, state):

    filepath = Path(get_original_cwd()).joinpath(cfg.input.local.filename)
    with xr.open_dataset(
        filepath
    ) as f:
        ds = f.load()

    if "time" in ds.dims:
        state.logger.info(
            f"Time dimension found. Selecting the first time step at {cfg.modules.time.start}"
        )
        ds = ds.sel(time=ds.time[cfg.modules.time.start])

    ds = xr.where(ds > 1e35, np.nan, ds)

    crop = np.any(list(dict(cfg.input.local.crop).values()))
    if crop:
        state.logger.info("Cropping dataset")
        ds = ds.sel(
            x=slice(cfg.input.local.crop.xmin, cfg.input.local.crop.xmax),
            y=slice(cfg.input.local.crop.ymin, cfg.input.local.crop.ymax),
        )

    max_coarsening_ratio = max(
        cfg.input.local.coarsening.ratio.x, cfg.input.local.coarsening.ratio.y
    )
    if max_coarsening_ratio > 1:
        state.logger.info("Coarsening dataset")
        ds = ds.coarsen(
            x=cfg.input.local.coarsening.ratio.x,
            y=cfg.input.local.coarsening.ratio.y,
            boundary=cfg.input.local.coarsening.boundary,
        ).mean()

    # Interpolating needed?
    # ds_test = ds.interp(x=2, method="linear")

    # Example on how to convert units with xarray! 'data' is an xarray dataset...
    # x = data.Geopotential_height_isobaric.metpy.x.metpy.convert_units('meter').values

    ds = complete_data(ds)

    for variable, array in ds.data_vars.items():
        setattr(state, variable, tf.Variable(array))

    for coord, array in ds.coords.items():
        setattr(
            state, coord, tf.constant(array)
        )  # constant for x, y, z, time etc. Should time be here though? What happens if its multiple time periods?

    # dt = xr.DataTree(name="root", dataset=ds)


def complete_data(ds: xr.Dataset) -> xr.Dataset:

    # ? Are units automatically included in some of these or not? I think no...
    X, Y = np.meshgrid(ds.x, ds.y)
    # ds["x"] = ds.x
    ds["dx"] = abs(ds.x.data[1] - ds.x.data[0])
    ds["dy"] = abs(ds.y.data[1] - ds.y.data[0])
    ds["X"] = xr.DataArray(X, dims=["y", "x"])
    ds["Y"] = xr.DataArray(Y, dims=["y", "x"])
    ds["dX"] = xr.DataArray(np.ones_like(X) * ds.dx.values, dims=["y", "x"])
    ds["dY"] = xr.DataArray(np.ones_like(Y) * ds.dy.values, dims=["y", "x"])
    ds["thk"] = xr.DataArray(np.zeros_like(X), dims=["y", "x"])

    topg_exists = "topg" in ds.data_vars
    usurf_exists = "usurf" in ds.data_vars
    if not topg_exists and not usurf_exists:
        raise ValueError("Either 'topg' or 'usurf' must be present in the dataset.")

    if "topg" not in ds.data_vars:
        # ? Should we only add pixels that do not have a nan value?
        ds["topg"] = ds["usurf"] - ds["thk"]
    elif "usurf" not in ds.data_vars:
        ds["usurf"] = ds["topg"] + ds["thk"]

    return ds
