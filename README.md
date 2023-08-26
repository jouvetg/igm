[![License badge](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
### <h1 align="center" id="title">The Instructed Glacier Model (IGM) -- 2.0 </h1>

# Overview    

The Instructed Glacier Model (IGM) is an **open-source Python package** and a **3D glacier evolution model**, which can simulate the coupling between ice thermo-dynamics, surface mass balance, mass concervation amon others. IGM features:

- **State-of-the-art physics:** IGM implements **high-order** 3D ice flow mechanics, an **Enthalpy** model, **melt/accumulation surface mass balance** model, and an increasing number of other glaciological processes.

- **Computational high efficiency:** Thanks to the **TensorFlow library**, mathematial operations are **vectorized**, and therefore run in parrallel. This permits tremendous **speed-ups on GPU**. **Physics-informed deep learning** is used as an alternative to solver for modelling ice flow physics in a parrallelized way. While GPU are highly recommended for large modelled domain, IGM runs fairly-well on CPU for individual glaciers.

- **Automatic differentiation:** TensorFlow operations permits Automatic Differentiaion, which strongly facilitate and speed-up inverse modelling / data assimilation, as well training of the ice flo neural network.

- **Simplicity and modularity:** IGM uses the most popular programming language -- Python -- at a relative low level of abstractivity. IGM is organized **module-wise** to facilitate coupling, user-customization and commmunity development. For simplicity, IGM assumes horizontal regular gridded discrtuzation and deals with **2D gridded input and output data**.


# Documentation

IGM's documentation can be found on the dedicated [wiki](https://github.com/jouvetg/igm2/wiki).
  
# Contact

Feel free to drop me an email for any questions, bug reports, or ideas of model extension: guillaume.jouvet at unil.ch

