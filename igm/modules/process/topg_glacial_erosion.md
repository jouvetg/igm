
### <h1 align="center" id="title">IGM module `topg_glacial_erosion` </h1>

# Description:

This IGM module implements change in basal topography (due to glacial erosion). The bedrock is updated (with a frequency provided by parameter `erosion_update_freq years`) assuming a power erosion law, i.e. the erosion rate is proportional (parameter `erosion_cst`) to a power (parameter `erosion_exp`) of the sliding velocity magnitude. 

By default, we use the parameters from
 
 ```
 Herman, F. et al., Erosion by an Alpine glacier. Science 350, 193-195, 2015.
```