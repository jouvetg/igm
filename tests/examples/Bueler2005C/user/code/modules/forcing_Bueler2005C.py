# Import the most important libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
 
def initialize(cfg,state):
    pass
 
def update(cfg,state):
    # compute time dependent mass balance according to Bueler C 2005
    A = 1.0e-16  # [Pa^-3 year^-1]
    n = 3.
    g = 9.81
    rho = 910.
    Gamma = 2.*A*(rho*g)**n / (n+2)

    # calculate a radius from the igm coordinate grids
    R = tf.sqrt(state.X**2. + state.Y**2.)

    ## calculate dome height for time step
    lambda_B = 5.
    H_0_B = 3600.
    R_0_B = 750000.
    t_0_B = 15208.
    alpha_B = (2.-(n+1.)*lambda_B)/(5.*n+3.)
    beta_B = (1.+(2.*n+1.)*lambda_B)/(5.*n+3.)

    H_B = tf.zeros(tf.shape(R))

    if state.t > 0.:
        H_B = H_0_B*(state.t/t_0_B)**(-alpha_B) * (1.-((state.t/t_0_B)**(-beta_B) * (R/R_0_B))**((n+1.)/n))**(n/(2.*n+1.))
        H_B = tf.where(tf.math.is_nan(H_B), 0., H_B)
        smbB = (5. / state.t) * H_B

    else:
        smbB = H_B


    state.smb  = smbB
    
def finalize(cfg,state):
    pass
