
### <h1 align="center" id="title">IGM module optimize </h1>

# Description:

This function does the data assimilation (inverse modelling) to optimize thk, 
slidingco and usurf from observational data from the follwoing reference:

@article{jouvet2023ice,
  title={Ice flow model emulator based on physics-informed deep learning},
  author={Jouvet, Guillaume and Cordonnier, Guillaume},
  year={2023},
  publisher={EarthArXiv}
}

# I/=

Input: usurfobs,uvelsurfobs,vvelsurfobs,thkobs, ...
Output: thk, slidingco, usurf
