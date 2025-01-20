
import numpy as np
import os, glob
import tensorflow as tf

from hydra.utils import get_original_cwd
from pathlib import Path

from igm.modules.utils import *

from .include_icemask import include_icemask
    
def run(cfg, state):
    import rasterio

    filepath = Path(get_original_cwd()).joinpath(cfg.input.load_tif.ltif_folder)

    files = glob.glob(os.path.join(filepath, "*.tif"))
    
    print(files)
    
    for file in files:
        var = os.path.split(file)[-1].split(".")[0]
        if os.path.exists(file):
            state.profile_tif_file = rasterio.open(file, "r").profile
            with rasterio.open(file) as src:
                vars()[var] = np.flipud(src.read(1))
                height = vars()[var].shape[0]
                width = vars()[var].shape[1]
                cols, rows = np.meshgrid(np.arange(width), np.arange(height))
                x, y = rasterio.transform.xy(src.transform, rows, cols)
                x = np.array(x)[0, :]
                y = np.flip(np.array(y)[:, 0])
            del src

    # coarsen if requested
    if cfg.input.load_tif.ltif_coarsen > 1:
        xx = x[:: cfg.input.load_tif.ltif_coarsen]
        yy = y[:: cfg.input.load_tif.ltif_coarsen]
        for file in files:
            var = os.path.split(file)[-1].split(".")[0]
            if (not var in ["x", "y"]) & (vars()[var].ndim == 2):
                vars()[var] = vars()[var][
                    :: cfg.input.load_tif.ltif_coarsen, :: cfg.input.load_tif.ltif_coarsen
                ]
        #                vars()[var] = RectBivariateSpline(y, x, vars()[var])(yy, xx) # does not work
        x = xx
        y = yy

    # crop if requested
    if cfg.input.load_tif.ltif_crop:
        i0 = max(0, int((cfg.input.load_tif.ltif_xmin - x[0]) / (x[1] - x[0])))
        i1 = min(int((cfg.input.load_tif.ltif_xmax - x[0]) / (x[1] - x[0])), x.shape[0] - 1)
        i1 = max(i0 + 1, i1)
        j0 = max(0, int((cfg.input.load_tif.ltif_ymin - y[0]) / (y[1] - y[0])))
        j1 = min(int((cfg.input.load_tif.ltif_ymax - y[0]) / (y[1] - y[0])), y.shape[0] - 1)
        j1 = max(j0 + 1, j1)
        #        i0,i1 = int((cfg.input.load_tif.ltif_xmin-x[0])/(x[1]-x[0])),int((cfg.input.load_tif.ltif_xmax-x[0])/(x[1]-x[0]))
        #        j0,j1 = int((cfg.input.load_tif.ltif_ymin-y[0])/(y[1]-y[0])),int((cfg.input.load_tif.ltif_ymax-y[0])/(y[1]-y[0]))
        for file in files:
            var = os.path.split(file)[-1].split(".")[0]
            if not var in ["x", "y"]:
                vars()[var] = vars()[var][j0:j1, i0:i1]
        y = y[j0:j1]
        x = x[i0:i1]

    # transform from numpy to tensorflow
    for file in files:
        var = os.path.split(file)[-1].split(".")[0]
        vars(state)[var] = tf.Variable(vars()[var].astype("float32"), trainable=False)

    state.x = tf.constant(x.astype("float32"))
    state.y = tf.constant(y.astype("float32"))

    complete_data(state)

    if cfg.input.load_tif.ltif_icemask_include:
        include_icemask(state, mask_shapefile=cfg.input.load_tif.ltif_icemask_shapefile, mask_invert=cfg.input.load_tif.ltif_icemask_invert)