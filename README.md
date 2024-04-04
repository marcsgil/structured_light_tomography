# structured-light-tomography

This repository contains the code used in the paper "Quantum tomography of structured light patterns from simple intensity measurements".

The code is divided in two main folders: 'Data' contains the experimental and simulated data used in the paper, as well as the code to generate the simulated data and treat the experimental data. 'Tomography' contains the code that outputs the predictions given the input data. It includes the training of the machine learning models. 

For future reuse, some code used in this paper has been turned into two separate packages: 
[BayesianTomography.jl](https://github.com/marcsgil/BayesianTomography.jl) and [PositionMeasurements.jl](https://github.com/marcsgil/PositionMeasurements.jl).