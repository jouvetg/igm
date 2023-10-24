
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