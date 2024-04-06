# structured-light-tomography

This repository contains the code used in the paper "Quantum tomography of structured light patterns from simple intensity measurements".

Relevant folders are: `Data` contains the code to generate the simulated data and treat the experimental data. If one wishes to reproduce the results of the paper, the experimental data is available [here](https://zenodo.org/records/10936150). The zip file should be extracted in the 'Data' folder, so that one could resolve the path `Data/Raw/mixed_intense.h5`, as an example. `Tomography` contains the code that outputs the predictions given the input data. It also includes the training of the machine learning models. 

For future reuse, some code used in this paper has been turned into two separate packages: 
[BayesianTomography.jl](https://github.com/marcsgil/BayesianTomography.jl) and [PositionMeasurements.jl](https://github.com/marcsgil/PositionMeasurements.jl).

Most of the code is written in [Julia](https://julialang.org/) (version 1.10), except for the machine learning part, which is written in [Python](https://www.python.org/) (version 3.10). The Julia dependencies can be downloaded simply by typing `]` in the REPL to enter in the Pkg mode, running `activate .` to activate the environment in this folder, followed finally by `instantiate` to download the necessary packages. The Python dependencies can be installed by running `pip install -r requirements.txt`. It is heavily suggested that you use a virtual environment.