
### <h1 align="center" id="title">IGM module optimize_v1 </h1>

# Description: 

A data assimilation module of IGM permits to seek optimal ice thickness, top ice surface, and ice flow parametrization (red variables in the following figure), that best explains observational data such as surface ice speeds, ice thickness profiles, top ice surface (blue variables in the following figure) while being consistent with the ice flow iflo_emulator used in forwrd modelling. This page explains how to use the data assimilation module as a preliminary step in IGM of a forward/prognostic model run. I recommend to read the paper reference given below before to proceed further. 

# Getting the data 

The first thing you need to do is to get as much data as possible, this includes:

* Observed surface ice velocities ${\bf u}^{s,obs}$, e.g. from Millan and al. (2022).
* Surface top elevation $s^{obs}$, e.g. SRTM, ESA GLO-30, ...
* Ice thickness profiles $h_p^{obs}$, e.g. GlaThiDa
* Glacier outlines, and resulting mask, e.g. from the Randolph Glacier Inventory.

Of course, you may not have all these data, which is fine. You work with a reduced amount of data, however, you will have make assumptions to reduce the number of variables to optimize (controls) to keep the optimization problem well-posed (i.e., with a unique solution).

All the data need to be assemblied in 2D raster grid in an netcdf file (called by default observation.nc) using convention variable names but ending with 'obs'. E.g. observation.nc contains fields 'usurfobs' (observed top surface elevation), 'thkobs' (observed thickness profiles, use nan or novalue where no data is available), 'icemaskobs' (this mask from RGI outline serve to enforce zero ice thickness outside the mask), 'uvelsurfobs' and 'vvelsurfobs' (x- and y- components of the horizontal surface ice velocity, use nan or novalue where no data is available), 'thkinit' (this may be a formerly-inferred ice thickness field to initalize the inverse model, otherwise it would start from thk=0).

**Use the IGM 'preprare_data' to download all the data you need using OGGM and the GlaThiDa database**

# Asumption on the ice flow control

Optimizing for both Arrhenius factor (A) and sliding coefficient (c) would lead to multiple solutions as several combinations of the two may explain the observed ice flow similarly. To deal with this issue, we introduce a single control of the ice flow strenght (named as 'strflowctrl' in IGM) $\tilde{A}$ = A + lambda c, where A is the Arrhenius factor that controls the ice shearing from cold-ice case (low A) to temperate ice case (A=78), c is a sliding coefficient that controls the strength of basal motion from no sliding (c=0) to high sliding (high c) and lambda=1 is a given parameter. 

# General optimization setting

The optimization problem consists of finding spatially varying fields ($h$, $\tilde{A}$, $s$) that minimize the cost function
$$ \mathcal{J}(h,\tilde{A},s) = \mathcal{C}^u + \mathcal{C}^h + \mathcal{C}^s + \mathcal{C}^{d} + \mathcal{R}^h +  \mathcal{R}^{\tilde{A}} +  \mathcal{P}^h, $$

where $\mathcal{C}^u$ is the misfit between modeled and observed surface ice velocities ($\mathcal{F}$ is the output of the ice flow iflo_emulator/neural iflo_network):
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
params.opti_control=['thk','strflowctrl','usurf'] # this is the most general case  
params.opti_control=['thk','usurf'] # this will only optimize ice thickness and top surface elevation
params.opti_control=['thk'] # this will only optimize ice thickness 
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
params.opti_velsurfobs_std = 5 # unit m/y
params.opti_thkobs_std     = 5 # unit m
params.opti_usurfobs_std   = 5 # unit m
params.opti_divfluxobs_std = 1 # unit m/y
```

Second, you may change regularization parameters such as i) $\alpha^h, \alpha^A$, which control the regularization weights for the ice thickness and strflowctrl (increasing $\alpha^h, \alpha^A$ will make thse fields spatially smoother), or ii) parameters beta and gamma involved for regularizing the ice thickness h. Taking beta=1 occurs to enforce isotropic smoothing, reducing beta will make the smoothing more and more anisotropic to enforce further smoothing along ice flow directions than accross directions (as expected for the topography of a glacier bedrock, which was eroded over long times). Setting parameter gamma to a small value may be usefull to add a bit of convexity in the system. This may help when initializing the inverse modelled with zero thickness, or to treat margin regions with no data available. These parameters may be changed as follows:

```python 
params.opti_regu_param_thk = 10.0            # weight for the regul. of thk
params.opti_regu_param_strflowctrl = 1.0     # weight for the regul. of strflowctrl
params.opti_smooth_anisotropy_factor = 0.2
params.opti_convexity_weight = 0.002
```

Lastly, there are a couple of other parameters we may be interest to change e.g.

```python 
params.opti_nbitmax       = 1000   # Number of it. for the optimization
params.opti_step_size     = 0.001  # step size in the optimization iterative algorithm
params.opti_init_zero_thk = True   # Force inializing with zero ice thickness (otherwise take thkinit)
```
 
# Monitoring the optimization

You may monitor the data assimilation during the inverse modelling in several ways:

* Check that the components of the costs decrease over time, the value of cost are printed during the optimization, and a graph is produced at the end.
* Set up glacier.config.plot_result = True and glacier.config.plt2d_live = True to monitor in live time the evolution of the field your are optimizing such as the ice thickness, the surface ice speeds, ect ... You may also check (hopefully decreasing) STD given in the figure.
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
