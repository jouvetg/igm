### <h1 align="center" id="title">IGM module `print_comp` </h1>

# Description:

This module reports the computational times taken by any IGM modules at the end of the model run directly in the terminal output, as well as in a file ("computational-statistics.txt"). It also produces a camember-like plot ( "computational-pie.png") displaying the relative importance of each module, computationally-wise. 

Note: These numbers must be interepreted with care: Leaks of computational times from one to another module are sometime observed (likely) due to asynchronous GPU calculations.
