[![License badge](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![CI badge](https://github.com/AdrienWehrle/earthspy/workflows/CI/badge.svg)](https://github.com/AdrienWehrle/igm/actions)
### <h1 align="center" id="title">IGM 2.0 </h1>

# Goal:
- Release of an improved version (IGM 2.0) that meets standard of collaborative codes, keeping it simple.
 
# Major change:
- Get rid of the all-in-one igm class, split into multiple indendent modules/files (functions) that may work independly
- igm is a python module that contains functions and utilities
- Make a proper independent parameter managers
- Keep the igm-run.py controlled by user.

# New concepts

- igm is a python module located in igm folder:

```
├── examples
│   └── aletsch-basic
├── igm
│   ├── __init__.py
│   ├── modules
│   │   ├── iceflow_v1.py
│   │   ├── load_ncdf_data.py
│   │   ├── load_tif_data.py
│   │   ├── ncdf_ex.py
│   │   ├── ncdf_ts.py
│   │   ├── plot_sp.py
│   │   ├── plot_vs.py
│   │   ├── prepare_data.py
│   │   ├── print_info.py
│   │   ├── smb_simple.py
│   │   ├── synthetic.py
│   │   ├── thk.py
│   │   ├── tif_ex.py
│   │   ├── time_step.py
│   │   └── utils.py
│   ├── params_core.py
│   └── state.py
├── LICENSE
├── model-lib
│   └── f15_cfsflow_GJ_22_a
```

- In igm-run.py one first define a suite of modules that will be called iteratively 
```python
modules = [
    "load_ncdf_data",  # this will read ncdf inpout data file
    "smb_simple",      # a simple surface mass balance
    "iceflow_v1",      # ice flow model component
    "time_step",       # compute time step and time
    "thk",             # mass conservation, dh/dt
    "ncdf_ex",         # outptut ncdf file on a regular basis
]
```

- params is a argparse set of parameters, parsing is done at the begining:
```python
parser = igm.params_core()
for module in modules:
    getattr(igm, "params_" + step)(parser)
params = parser.parse_args()
```

- state is a variable that contains all "state" variables, e.g. state.thk permits to access ice thickness, it replaces the former glacier, but without the functions.
```python
state = igm.State(params)
```

- igm-run.py defines in turn params, and state, and then iterate over all modules 
```python
with tf.device("/GPU:0"):
    # Initialize all the model components in turn
    for module in [m for m in modules if hasattr(igm, "init_" + s)]:
        getattr(igm, "init_" + step)(params, state)

    # Time loop, perform the simulation until reaching the defined end time
    while state.t < params.tend:
        # Update in turn each model components
        for for module in [m for m in modules if hasattr(igm, "update_" + s)]:
            getattr(igm, "update_" + step)(params, state)
```
 
