import igm
import tensorflow as tf
import pytest
import numpy as np
from make_fake_ncdf import write_ncdf
import os
 
def test_load_ncdf():
    
    write_ncdf()
 
    state = igm.State()

    state.original_cwd = ""

    cfg = igm.EmptyClass()  
    cfg.inputs  = igm.load_yaml_as_cfg(os.path.join("conf","inputs","load_ncdf.yaml"))
 
    igm.inputs.load_ncdf.run(cfg, state)

    ny,nx = state.thk.shape

    mid = state.topg[int(ny/2),int(nx/2)]
 
    assert (mid>2450)&(mid<2550)

    for f in ['input.nc']:
        if os.path.exists(f):
            os.remove(f)
