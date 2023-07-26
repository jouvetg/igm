
argmark
=======

# Usage:


```bash
usage: argmark [-h] [--tstart TSTART] [--tend TEND] [--tsave TSAVE] [--cfl CFL] [--dtmax DTMAX]

```
# Arguments

|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--tstart`|`2000.0`|Start modelling time (default 2000)|
||`--tend`|`2100.0`|End modelling time (default: 2100)|
||`--tsave`|`10`|Save result each X years (default: 10)|
||`--cfl`|`0.3`|CFL number for the stability of the mass conservation scheme,         it must be below 1 (Default: 0.3)|
||`--dtmax`|`10.0`|Maximum time step allowed, used only with slow ice (default: 10.0)|
