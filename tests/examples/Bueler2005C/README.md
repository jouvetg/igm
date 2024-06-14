
Test written by Alexander Jarosh based on Ed Bueler's work on SIA flow model benchmarks from 2005:

Bueler E, Lingle CS, Kallen-Brown JA, Covey DN, Bowman LN. Exact solutions and verification of numerical models for isothermal ice sheets. Journal of Glaciology. 2005;51(173):291-306. doi:10.3189/172756505781829449

I have added the Bueler C test (Table 2, p. 297 in his paper) as an example to IGM to see if IGM would grow a circular ice sheet, driven by circular symmetric, time evolving surface mass balance. That benchmark would run for a very long time (about 15000 years) but we already see issues in the first 1000 years. So I thought I stop there and submit the code.
The setup used here is based on the EISMINT I experiment parameters, which mostly are also used in IGM. I just set the sliding coeff to 0 (to avoid sliding) and A = 100 [MPa^-3 years^-1] coming from the A = 1x10^-16 [Pa^-3 years^-1] used in EISMINT, in Ed's work and in my numerical reference surface from a SIA model.

The numerical reference SIA surface is created with a mass conserving SIA model:
Jarosch, A. H., Schoof, C. G., and Anslow, F. S.: Restoring mass conservation to shallow ice flow models over complex terrain, The Cryosphere, 7, 229–240, doi:10.5194/tc-7-229-2013, 2013.

----- This is work in progress as the test does not work yet.

- First results have shown  Directional flow preference in IGM, demonstrated by Ed Bueler's C (2005) dome benchmark with time evolving smb (PR #22)

- This was partially fixed by increasing the amount of training, however, strictly not in the case of no-slip cndition, this has to be further investigated

- The symertry is pefectly preserved when using the solver

- The SMB must be checked and tested against the analtical solution and the SIA one of Alexander Jarosh
