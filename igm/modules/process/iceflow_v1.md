

### <h1 align="center" id="title">IGM module `ice_flow_v1` </h1>

# Description:

This IGM module models ice flow using a Convolutional Neural Network
following the former online training from external data.

You may find trained and ready-to-use ice flow emulators in the folder
`emulators/T_M_I_Y_V/R/`, where 'T_M_I_Y_V' defines the iflo_emulator, and
R defines the spatial resolution. Make sure that the resolution of the
picked iflo_emulator is available in the database. Results produced with IGM
will strongly rely on the chosen iflo_emulator. Make sure that you use the
iflo_emulator within the hull of its training dataset (e.g., do not model
an ice sheet with an iflo_emulator trained with mountain glaciers) to ensure
reliability (or fidelity w.r.t to the instructor model) -- the iflo_emulator
is probably much better at interpolating than at extrapolating.
Information on the training dataset is provided in a dedicated README
coming along with the iflo_emulator.

At the time of writing, I recommend using *f15_cfsflow_GJ_22_a*, which
takes ice thickness, top surface slopes, the sliding coefficient c
('slidingco'), and Arrhenuis factor A ('arrhenius'), and return basal,
vertical-average and surface x- and y- velocity components.

I have trained *f15_cfsflow_GJ_22_a* using a large dataset of modeled
glaciers (based on a Stokes-based CfsFlow ice flow solver) and varying
sliding coefficient c, and Arrhenius factor A into a 2D space.

It takes as inputs (thk, usurf, arrhenuis, slidingco) and provides
as output: (ubar,vbar, uvelsurf, vvelsurf, uvelbase, vvelbase)
