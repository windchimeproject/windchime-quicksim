# Simulation of Windchime and Sensitivity Calculation

This code simulates Windchime using large matrix operations and computes its sensitivity.  The main logic is in 'windchime.py'.  

It uses multiple optimization tricks:

* Do everything as a numpy operation for matrix operations at C speed
* 'Histogramming' to lower the precision of certain numbers to avoid excessive memory use, which can be downsamples for visualization
* (TODO: Use jax.numpy to run this on a GPU, and also replace the 'for' loop with something parallelizable such as 'vmap'

Jupyter notebooks are in analysis_notebooks for a demo and also (soon) paper plots.

(Aside: charles_end_of_summer_work is Charles' work at the end of summer, which has to be merged with the rest.)

