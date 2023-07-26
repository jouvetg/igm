
argmark
=======

# Usage:


```bash
usage: argmark [-h] [--RGI RGI] [--preprocess PREPROCESS] [--dx DX] [--border BORDER]
               [--thk_source THK_SOURCE] [--include_glathida INCLUDE_GLATHIDA]
               [--path_glathida PATH_GLATHIDA] [--output_geology OUTPUT_GEOLOGY]

```
# Arguments

|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--RGI`|`RGI60-11.01450`|RGI ID|
||`--preprocess`||Use preprocessing|
||`--dx`|`100`|Spatial resolution (need preprocess false to change it)|
||`--border`|`30`|Safe border margin  (need preprocess false to change it)|
||`--thk_source`|`consensus_ice_thickness`|millan_ice_thickness or consensus_ice_thickness in geology.nc|
||`--include_glathida`||Make observation file (for IGM inverse)|
||`--path_glathida`|`/home/gjouvet/`|Path where the Glathida Folder is store, so that you don't need               to redownload it at any use of the script|
||`--output_geology`||Write prepared data into a geology file|
