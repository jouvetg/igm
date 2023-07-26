
### <h1 align="center" id="title">IGM module optimize_v1 </h1>

# Description:

This function does the data assimilation (inverse modelling) to optimize thk, 
strflowctrl and usurf from observational data from the follwoing reference:

@article{jouvet2023inversion,
  title={Inversion of a Stokes glacier flow model emulated by deep learning},
  author={Jouvet, Guillaume},
  journal={Journal of Glaciology},
  volume={69},
  number={273},
  pages={13--26},
  year={2023},
  publisher={Cambridge University Press}
}

# I/O

Input: usurfobs,uvelsurfobs,vvelsurfobs,thkobs, ...
Output: thk, strflowctrl, usurf
