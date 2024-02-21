# structured-light-tomography

This repository contains the code used in the paper "Machine Learning enhanced tomography of structured light qudits".

The code is divided in two main folders: 'Data' contains the experimental and simulated data used in the paper, as well as the code to generate the simulated data and treat the experimental data. 'Tomography' contains the code that outputs the predictions given the input data. It includes the training of the machine learning models. The other methods are implemented in the [BayesianTomography.jl](https://github.com/marcsgil/BayesianTomography.jl) package.