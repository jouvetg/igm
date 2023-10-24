
### <h1 align="center" id="title">IGM module `glerosion` </h1>

# Description:

This IGM module implements change in basal topography (due to glacial erosion). The bedrock is updated (with a frequency provided by parameter `glerosion_update_freq years`) assuming a power erosion law, i.e. the erosion rate is proportional (parameter `glerosion_cst`) to a power (parameter `glerosion_exp`) of the sliding velocity magnitude. 

By default, we use the parameters from
 
 ```
 Herman, F. et al., Erosion by an Alpine glacier. Science 350, 193-195, 2015.
``` 
# Parameters: 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--glerosion_cst`|`2.7e-07`|Erosion multiplicative factor, here taken from Herman, F. et al.               Erosion by an Alpine glacier. Science 350, 193–195 (2015)|
||`--glerosion_exp`|`2`|Erosion exponent factor, here taken from Herman, F. et al.                Erosion by an Alpine glacier. Science 350, 193–195 (2015)|
||`--glerosion_update_freq`|`1`|Update the erosion only each X years (Default: 100)|
