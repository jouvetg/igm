
### <h1 align="center" id="title">IGM module `thk` </h1>

# Description:

This IGM module solves the mass conservation of ice to update the thickness from ice flow (computed from module `iceflow`) and surface mass balance (given any module that update `smb`). The mass conservation equation is solved using an explicit first-order upwind finite-volume scheme on the 2D working grid. With this scheme mass of ice is allowed to move from cell to cell (where thickness and velocities are defined) from edge-defined fluxes (inferred from depth-averaged velocities, and ice thickness in upwind direction). The resulting scheme is mass conservative and parallelizable (because fully explicit). However, it is subject to a CFL condition. This means that the time step (defined in module `time`) is controlled by parameter parameter `time_cfl`, which is the maximum number of cells crossed in one iteration (this parameter cannot exceed one), see the documentation of module `time`. A bit more details on the scheme are given in the following paper.

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
 