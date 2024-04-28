import igm
import tensorflow as tf
import pytest
import numpy as np
from make_fake_ncdf import write_ncdf
import os

write_ncdf()
 
state = igm.State()
modules_dict = {'modules_preproc': ['load_ncdf'], 'modules_process': [], 'modules_postproc': []}
 
imported_modules = igm.load_modules(modules_dict)
 
module = imported_modules[0] 

parser = igm.params_core()

module.params(parser)

params, __ = parser.parse_known_args()
 
module.initialize(params, state)

module.update(params, state)

module.finalize(params, state)

ny,nx = state.thk.shape

mid = state.topg[int(ny/2),int(nx/2)]
 
assert (mid>2450)&(mid<2550)

for f in ['input.nc']:
    if os.path.exists(f):
        os.remove(f)