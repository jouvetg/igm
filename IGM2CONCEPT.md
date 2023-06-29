[![License badge](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![CI badge](https://github.com/AdrienWehrle/earthspy/workflows/CI/badge.svg)](https://github.com/AdrienWehrle/igm/actions)

### <h1 align="center" id="title">IGM 2.0 -- concept </h1>

# Goal:
- Release of an improved version (IGM 2.0) that meets standard of collaborative codes, keeping it simple.
 
# Major change:
- Get rid of the all-in-one igm class, split into multiple indendent modules/files (functions) that may work independly
- Make igm a python module that contains functions and utilities
- Proper and independent parameter manager

# Basic about glacier evolution modelling   

IGM as any other glacier evolution models  simulates the ice dynamics, surface mass balance, and its coupling through mass conservation to predict the evolution of glaciers, icefields, or ice sheets (Figs. 1 and 2). For computational efficiency, IGM models the ice flow by a Neural Network, which is trained with state-of-the-art ice flow models (Fig. 3).

![Alt text](./fig/cores-figs.png)

Therefore, IGM does in turn essentially two things:
- Reading spatially distributed variables (e.g. surface topography, ice thickness) or infer them from observations.
- A time loop over the desired period updating in turn surface mass balance, ice flow, ice thickness, ....

From this basic concept, one can make the model more sophisticated by adding model components (climate params, lithospheric processes,...)

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

- Each file XXX of folder 'module' contains a suite of functions. The most important are
params_XXX(parser), init_XXX(params,state) and update_XXX(params,state), which provide
the parameters, initialize and update the quantity XXX within the time iteration. E.g.
'smb_simple.py' contains functions params_smb_simple(parser), init_smb_simple(params,state)
and update_smb_simple(params,state).

- In igm-run.py, one first defines a suite of modules that will be called iteratively later on
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
In the above list, the user is free to add any other existing or own-made modules
(e.g. to compute particle trajectories, to compute ice temperature, ploting, reading
different format like tif, printing live informations, ect..)

- Then, params is a argparse set of parameters, parsing is done at the begining
(only the parameters of the module list are called):
```python
parser = igm.params_core()
for module in modules:
    getattr(igm, "params_" + step)(parser)
params = parser.parse_args()
```

- state is a variable that contains all "state" variables, e.g. state.thk permits to 
access ice thickness, it replaces the former glacier, but without the functions.
```python
state = igm.State(params)
```
state is now nearly empty, but contains a few technical features (working path, logging, 
track of computational time, ...)

- The core code of igm-run.py is to set the computation on a device (GPU or CPU),
initialize all modules in turn, and then do a time loop of update of all modules: 
```python
with tf.device("/GPU:0"):
    # Initialize all the model components in turn
    for module in modules:
        getattr(igm, "init_" + step)(params, state)

    # Time loop, perform the simulation until reaching the defined end time
    while state.t < params.tend:
        # Update in turn each model components
        for for module in modules:
            getattr(igm, "update_" + step)(params, state)
```
 If modules=["smb_simple","iceflow_v1","time_step","thk"], then the above code is equivalent to 
```python
with tf.device("/GPU:0"):
    init_smb_simple(params, state)
    init_iceflow_v1(params, state)
    init_time_step(params, state)
    init_thk(params, state)
 
    while state.t < params.tend:
        update_smb_simple(params, state)
        update_iceflow_v1(params, state)
        update_time_step(params, state)
        update_thk(params, state)
```