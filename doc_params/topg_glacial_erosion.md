
argmark
=======

# Usage:


```bash
usage: argmark [-h] [--erosion_cst EROSION_CST] [--erosion_exp EROSION_EXP]
               [--erosion_update_freq EROSION_UPDATE_FREQ]

```
# Arguments

|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--erosion_cst`|`2.7e-07`|Erosion multiplicative factor, here taken from Herman, F. et al.               Erosion by an Alpine glacier. Science 350, 193–195 (2015)|
||`--erosion_exp`|`2`|Erosion exponent factor, here taken from Herman, F. et al.                Erosion by an Alpine glacier. Science 350, 193–195 (2015)|
||`--erosion_update_freq`|`1`|Update the erosion only each X years (Default: 100)|
