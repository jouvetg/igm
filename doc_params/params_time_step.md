usage: PAR.py [-h] [--tstart TSTART] [--tend TEND] [--tsave TSAVE] [--cfl CFL]
              [--dtmax DTMAX]

optional arguments:
  -h, --help       show this help message and exit
  --tstart TSTART  Start modelling time (default 2000)
  --tend TEND      End modelling time (default: 2100)
  --tsave TSAVE    Save result each X years (default: 10)
  --cfl CFL        CFL number for the stability of the mass conservation
                   scheme, it must be below 1 (Default: 0.3)
  --dtmax DTMAX    Maximum time step allowed, used only with slow ice
                   (default: 10.0)
