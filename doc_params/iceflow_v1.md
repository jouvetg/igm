
argmark
=======

# Usage:


```bash
usage: argmark [-h] [--init_strflowctrl INIT_STRFLOWCTRL] [--emulator EMULATOR]
               [--init_slidingco INIT_SLIDINGCO] [--init_arrhenius INIT_ARRHENIUS]
               [--multiple_window_size MULTIPLE_WINDOW_SIZE] [--force_max_velbar FORCE_MAX_VELBAR]

```
# Arguments

|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--init_strflowctrl`|`78`|Initial strflowctrl (default 78)|
||`--emulator`|`f15_cfsflow_GJ_22_a`|Directory path of the deep-learning ice flow model,               create a new if empty string|
||`--init_slidingco`|`0`|Initial sliding coeeficient slidingco (default: 0)|
||`--init_arrhenius`|`78`|Initial arrhenius factor arrhenuis (default: 78)|
||`--multiple_window_size`|`0`|If a U-net, this force window size a multiple of 2**N (default: 0)|
||`--force_max_velbar`|`0`|This permits to artif. upper-bound velocities, active if > 0 (default: 0)|
