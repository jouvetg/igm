[![License badge](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![CI badge](https://github.com/AdrienWehrle/earthspy/workflows/CI/badge.svg)](https://github.com/AdrienWehrle/igm/actions)
### <h1 align="center" id="title">IGM 2.0 </h1>

# Goal:
- Release of an improved version (IGM 2.0) that meets standard of collaborative codes.

# Requirements:
- Keeping IGM simple to use! 
- Meet coding standards
- Keep compatibility with IGM 1
- ????? compatibility with CSDMS , OGGM, G. Cordonnier’s code
 
# Major change:
- Get rid of the all-in-one igm class, split into multiple indendent modules/files (functions) that may work independly
- igm is a python module that contains functions and utilities
- Make a proper independent parameter managers
- Keep the igm-run.py controlled by user.

# New concepts

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
 
# Model components

- Mass conservation
- Surface mass balance (simple + PDD-like)
- High-order ice dynamics (based on “PINN”)
- Lagrangian particle tracking
- Glacial Erosion
- Lithospheric loading (J. Mey)
- Snow Avalanche (J. Mey)
- Enthalpy model for ice (currently implemented with numpy/numpy)
- Shelf & calving front
- Fluvial erosion (GC/BF)
- Rock avalanches (GC/BF)
- Sub-glacial hydrology (S. Cook project

# General optimization setting

The optimization problem consists of finding spatially varying fields ($h$, $\tilde{A}$, $s$) that minimize the cost function
$$ \mathcal{J}(h,\tilde{A},s) = \mathcal{C}^u + \mathcal{C}^h + \mathcal{C}^s + \mathcal{C}^{d} + \mathcal{R}^h +  \mathcal{R}^{\tilde{A}} +  \mathcal{P}^h, $$

where $\mathcal{C}^u$ is the misfit between modeled and observed surface ice velocities ($\mathcal{F}$ is the output of the ice flow emulator/neural network):
$$ \mathcal{C}^u = \int_{\Omega} \frac{1}{2 \sigma_u^2} \left| {\bf u}^{s,obs} - \mathcal{F}( h, \frac{\partial s}{\partial x}, \frac{\partial s}{\partial y}, \tilde{A})  \right|^2,  $$

where $\mathcal{C}^h$ is the misfit between modeled and observed ice thickness profiles:
$$ \mathcal{C}^h = \sum_{p=1,...,P} \sum_{i=1,...,M_p} \frac{1}{2 \sigma_h^2}  | h_p^{obs}  (x^p_i, y^p_i) - h (x^p_i, y^p_i) |^2, $$

where $\mathcal{C}^s$ is the misfit between the modeled and observed top ice surface:
$$ \mathcal{C}^s = \int_{\Omega} \frac{1}{2 \sigma_s^2}  \left| s - s^{obs}  \right|^2,$$

where $\mathcal{C}^{d}$ is a misfit term between the flux divergence and its polynomial 
regression $d$ with respect to the ice surface elevation $s(x,y)$ to enforce smoothness with  dependence to $s$:
$$ \mathcal{C}^{d} = \int_{\Omega} \frac{1}{2 \sigma_d^2} \left| \nabla \cdot (h {\bar{\bf u}}) - d  \right|^2, $$

where $\mathcal{R}^h$ is a regularization term to enforce anisotropic smoothness and convexity of $h$:
$$ \mathcal{R}^h = \alpha_h \int_{h>0} \left(  | \nabla h \cdot \tilde{{\bf u}}^{s,obs} |^2 + \beta  | \nabla h \cdot (\tilde{{\bf u}}^{s,obs})^{\perp} |^2   -  \gamma h  \right),  $$

where $\mathcal{R}^{\tilde{A}}$ is a regularization term to enforce smooth A:
$$ \mathcal{R}^{\tilde{A}} = \alpha_{\tilde{A}} \int_{\Omega} | \nabla  \tilde{A}  |^2, $$

where $\mathcal{P}^h$ is a penalty term to enforce nonnegative ice thickness, and zero thickness outside a given mask:
$$ \mathcal{P}^h  = 10^{10} \times \left( \int_{h<0} h^2 + \int_{\mathcal{M}^{\rm ice-free}} h^2 \right).$$

Check at the reference paper given below for more explanation on the regularization terms.

# Define controls and cost components

The above optimization problem is given in the most general case, however, you may select only some components according to your data as follows: 

* the list of control variables you wish to optimize, e.g.
```python
glacier.config.opti_control=['thk','strflowctrl','usurf'] # this is the most general case  
glacier.config.opti_control=['thk','usurf'] # this will only optimize ice thickness and top surface elevation
glacier.config.opti_control=['thk'] # this will only optimize ice thickness 
```
* the list of cost components you wish to minimize, e.g.
```python
glacier.config.opti_cost=['velsurf','thk','usurf','divfluxfcz','icemask']  # this is the most general case  
glacier.config.opti_cost=['velsurf','icemask']  # In this case, you only fit surface velocity and ice mask.
```
*Make sure you have a balance between controls and constraints to ensure the problem to have a unique solution.*

# Exploring parameters

There are parameters that may need to tune for each application.

First, you may change your expected confidence levels (i.e. tolerance to fit the data) $\sigma^u, \sigma^h, \sigma^s, \sigma^d$ to fit surface ice velocity, ice thickness, surface top elevation, or divergence of the flux as follows:

```python
glacier.config.opti_velsurfobs_std = 5 # unit m/y
glacier.config.opti_thkobs_std     = 5 # unit m
glacier.config.opti_usurfobs_std   = 5 # unit m
glacier.config.opti_divfluxobs_std = 1 # unit m/y
```

Second, you may change regularization parameters such as i) $\alpha^h, \alpha^A$, which control the regularization weights for the ice thickness and strflowctrl (increasing $\alpha^h, \alpha^A$ will make thse fields spatially smoother), or ii) parameters beta and gamma involved for regularizing the ice thickness h. Taking beta=1 occurs to enforce isotropic smoothing, reducing beta will make the smoothing more and more anisotropic to enforce further smoothing along ice flow directions than accross directions (as expected for the topography of a glacier bedrock, which was eroded over long times). Setting parameter gamma to a small value may be usefull to add a bit of convexity in the system. This may help when initializing the inverse modelled with zero thickness, or to treat margin regions with no data available. These parameters may be changed as follows:

```python 
glacier.config.opti_regu_param_thk = 10.0            # weight for the regul. of thk
glacier.config.opti_regu_param_strflowctrl = 1.0     # weight for the regul. of strflowctrl
glacier.config.opti_smooth_anisotropy_factor = 0.2
glacier.config.opti_convexity_weight = 0.002
```

Lastly, there are a couple of other parameters we may be interest to change e.g.

```python 
glacier.config.opti_nbitmax       = 1000   # Number of it. for the optimization
glacier.config.opti_step_size     = 0.001  # step size in the optimization iterative algorithm
glacier.config.opti_init_zero_thk = True   # Force inializing with zero ice thickness (otherwise take thkinit)
glacier.config.observation_file   = 'observation.nc'
```

# Running the optimization

The optimization scheme is implemented in igm function optimize(), calling it for inverse modelling would look like this:

```python 
import numpy as np
import tensorflow as tf

from igm import Igm

glacier = Igm() 
 
# change parameters
glacier.config.iceflow_model_lib_path='../../model-lib/f14_pismbp_GJ_21_a' 
glacier.config.opti_control=['thk','strflowctrl','usurf']
glacier.config.opti_cost=['velsurf','thk','usurf','divfluxfcz','icemask']   
glacier.config.opti_usurfobs_std             = 5.0   # Tol to fit top ice surface 
glacier.config.plot_result           = True
glacier.config.plot_live             = True

glacier.initialize()

with tf.device(glacier.device_name):
    glacier.load_ncdf_data(glacier.config.observation_file)
    glacier.initialize_fields()
    glacier.optimize()
    
glacier.print_all_comp_info()
```

# Monitoring the optimization

You may monitor the data assimilation during the inverse modelling in several ways:

* Check that the components of the costs decrease over time, the value of cost are printed during the optimization, and a graph is produced at the end.
* Set up glacier.config.plot_result = True and glacier.config.plot_live = True to monitor in live time the evolution of the field your are optimizing such as the ice thickness, the surface ice speeds, ect ... You may also check (hopefully decreasing) STD given in the figure.
* You may do the same monitoring after the run looking at optimize.nc
* If you asked divfluxfcz to be in glacier.config.opti_cost, you should check what look like the divergence of the fluc (divflux)

# Reference

	@article{IGM-inv,
	  author       = "Jouvet, G.",
	  title        = "Inversion of a Stokes ice flow model emulated by deep learning",
	  DOI          = "10.1017/jog.2022.41",
	  journal      = "Journal of Glaciology",
	  year         = "2022",
	  pages        = "1--14",
	  publisher    = "Cambridge University Press"
	}
