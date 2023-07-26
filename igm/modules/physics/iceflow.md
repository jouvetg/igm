### <h1 align="center" id="title">IGM module iceflow </h1>

# Description:

This IGM module models ice flow using a Convolutional Neural Network based on 
Physics Informed Neural Network. You may find pre-trained and ready-to-use ice 
flow emulators, e.g. using the default emulator = f21_pinnbp_GJ_23_a, or using 
an initial untrained with emulator =''.

# I/O

Input: thk, usurf, arrhenuis, slidingco, dX
Output: U, V

# Parameters

!include https://github.com/jouvetg/igm2/blob/main/doc_params/optimize.md