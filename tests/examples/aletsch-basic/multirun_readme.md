### I explain here how we can do a mulitrun experiment on many gpus

Lets say we are on a node that has 4 gpus, using (`nvidia-smi`):

IMAGE

Instead of running a single experiment from 1 param file like done previously, we can run multiple experiments

# Single param file, multirun from command line arguments
# Multiple param files, simple command line (no arguments)
# Sequential experiments
# Parallel Experiments

# Changing folder/experiment names
# Terminal output
# Data locations

<!-- igm_run +experiment=params hydra.job.chdir=True core.hardware.visible_gpus=[0],[1] -->