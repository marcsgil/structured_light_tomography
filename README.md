# structured-light-tomography

This repository contains the code used in the paper "Tomography of the spatial structure of light".

Relevant folders are: `Data` contains the code to generate the simulated data and treat the experimental data. If one wishes to reproduce the results of the paper, the experimental data is available [here](https://zenodo.org/records/14002229). The zip file should be extracted in the 'Data' folder, so that one could resolve the path `Data/Raw/mixed_intense.h5`, as an example. `Tomography` contains the code that outputs the predictions given the input data.

For future reuse, some code used in this paper has been turned into two separate packages: 
[QuantumMeasurements.jl](https://github.com/marcsgil/QuantumMeasurements.jl).

The code is written in [Julia](https://julialang.org/) (version 1.11). The dependencies can be downloaded simply by typing `]` in the REPL to enter in the Pkg mode, running `activate .` to activate the environment in this folder, followed finally by `instantiate` to download the necessary packages.