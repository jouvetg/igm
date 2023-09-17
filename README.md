[![License badge](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
### <h1 align="center" id="title">The Instructed Glacier Model (IGM) -- 2.0 </h1>

# Overview    

The Instructed Glacier Model (IGM) is an **open-source Python package**, which permits to simulate **3D glacier evolution** accounting for the coupling between ice thermo-dynamics, surface mass balance, and mass conservation. IGM features:

- **Simplicity and modularity:** IGM is implemented in the most popular programming language -- Python -- at a low level of abstractivity. IGM is organized **module-wise** for clarity and to facilitate coupling, customization and commmunity development. For simplicity, IGM assumes a horizontal regular grid for numerical discretization and therefore deals with **2D gridded input and output data**.

- **State-of-the-art physics:** IGM implements mass conservation, **high-order** 3D ice flow mechanics, an **Enthalpy** model for the thermic regime of ice, **melt/accumulation surface mass balance** model, and other glaciological processes.

- **Computational high efficiency:** Thanks to the **TensorFlow library**, mathematical operations are **fully-vectorized**. This permits tremendous **speed-ups on GPU**. **Physics-informed deep learning** is used as an alternative to numerical solvers for modelling ice flow physics in a vectorized way. While GPU are highly-recommended for modelling large domain / high resolution, IGM runs fairly well on CPU for individual glaciers.

- **Automatic differentiation:** TensorFlow operations are differentiable. Therefore, automatic differentiation strongly facilitates and speeds-up inverse modelling / data assimilation.
  
# Documentation    

**IGM's documentation can be found on the dedicated [wiki](https://github.com/jouvetg/igm/wiki).**
  
# Contact

Feel free to drop me an email for any questions, bug reports, or ideas of model extension: guillaume.jouvet at unil.ch

