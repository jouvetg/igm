
### <h1 align="center" id="title">IGM module `time` </h1>

# Description:

This IGM modules computes time step such that i) it satisfy the CFL condition (controlled by parameter `clf`) ii) it is lower than a given maximum time step (controlled by parameter `time_step_max`) iii) it hits exactly given saving times (controlled by parameter `time_save`). The module additionally updates the time $t$ in addition to the time step.

Indeed, for stability reasons of the transport scheme for the ice thickness evolution, the time step must respect a CFL condition, controlled by parameter `time_cfl`, which is the maximum number of cells crossed in one iteration (this parameter cannot exceed one). By default, we take `time_cfl` to 0.3. We additionally request time step to be upper-bounded by a user-defined parameter `time_save` (default: 1 year).
 
Among the parameters of this module `time_start` and `time_end` defines the simulation starting and ending times, while `time_save` defines the frequency at which results must be saved (default: 10 years).

A bit more details on the time step stability conditionsis given in the following paper.

```
@article{jouvet2022deep,
  author =        {Jouvet, Guillaume and Cordonnier, Guillaume and
                   Kim, Byungsoo and L{\"u}thi, Martin and
                   Vieli, Andreas and Aschwanden, Andy},
  journal =       {Journal of Glaciology},
  number =        {270},
  pages =         {651--664},
  publisher =     {Cambridge University Press},
  title =         {Deep learning speeds up ice flow modelling by several
                   orders of magnitude},
  volume =        {68},
  year =          {2022},
  doi =           {10.1017/jog.2021.120},
}
``` 
# Parameters: 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--working_dir`|``|Working directory (default empty string)|
||`--modules_preproc`|`['oggm_shop']`|List of pre-processing modules|
||`--modules_process`|`['iceflow', 'time', 'thk']`|List of processing modules|
||`--modules_postproc`|`['write_ncdf', 'plot2d', 'print_info']`|List of post-processing modules|
||`--logging`||Activate the looging|
||`--logging_file`|``|Logging file name, if empty it prints in the screen|
||`--print_params`||Print definitive parameters in a file for record|
||`--time_start`|`2000.0`|Start modelling time|
||`--time_end`|`2100.0`|End modelling time|
||`--time_save`|`10`|Save result frequency for many modules (in year)|
||`--time_cfl`|`0.3`|CFL number for the stability of the mass conservation scheme, it must be below 1|
||`--time_step_max`|`1.0`|Maximum time step allowed, used only with slow ice|
