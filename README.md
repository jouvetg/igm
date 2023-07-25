[![License badge](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
### <h1 align="center" id="title">The Instructed Glacier Model (IGM) -- 2.0 </h1>

# Overview   

IGM as any other glacier evolution models simulates the ice dynamics, surface mass balance, and its coupling through mass conservation to predict the evolution of glaciers, icefields, or ice sheets (Figs. 1 and 2). The specificity of IGM is that it models the ice flow by a Neural Network, which is trained from ice flow physics (Fig. 3). As a result, IGM permits user-friendly, highly efficient, and mechanically state-of-the-art glacier simulations.

![Alt text](./fig/cores-figs.png)

Therefore, IGM does in turn essentially 3 things: i) pre-processing by preparing/reading spatially distributed variables (e.g. surface topography, ice thickness) or infer them from observations ii) a time loop over the desired period updating in turn surface mass balance, ice flow, ice thickness, ... iii) post-processing to nicely render the modelling results. IGM is a modular open-source Python package, which runs across both CPU and GPU and deals with two-dimensional gridded input and output data. 
  
# Manual / Wiki

IGM's documentation can be found on the dedicated [documentation](https://github.com/jouvetg/igm2/wiki)  
  
# Quick start

The easiest and quickest way is to get to know IGM is to run notebooks in [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jouvetg/igm/), which offers free access to GPU, or to install IGM on your machine, and start with examples.

# Contact

Feel free to drop me an email for any questions, bug reports, or ideas of model extension: guillaume.jouvet at unil.ch

