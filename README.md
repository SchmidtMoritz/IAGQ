# IAGQ

Master thesis: Implementation and analysis of gradient-computation in quantum-algorithms

## Installation

For this tool [poetry](https://python-poetry.org/) is used as a python packaging tool.

Install poetry and run:

`poetry install`

in the main folder to create a virtual environment including all required python libraries.

## Scripts

### plots_script.py

* Generates all plots used in the thesis from scratch. Takes some time!

### Experiment scripts

There are also four scripts provided to run different experiments.

#### gradient_sample_shotslist.py

* Computes error of gradient-approximations with forward, backward and central finite differences, SPSA as well as general and standard parameter-shift for a given list of shots run on a simulator.

#### gradient_variance_shotslist.py

* Computes expected error of gradient-approximations with forward, backward and central finite differences, SPSA as well as general and standard parameter-shift for a given list of shots by computing the exact one-shot variance of all methods.

#### FD_variance_shotslist.py

* Computes expected error of gradient-approximations central finite differences with different shift values for a given list of shots by computing the exact one-shot variance of all methods.


* Use the argument `-cd_h` to specify distances $h$ to be used in the experiments, for example: `-cd_h 0.5 0.1 0.001` 
#### GPS_variance_shotslist.py

* Computes expected error of gradient-approximations general parameter-shift with different shift values as well as standard parameter-shift for a given list of shots by computing the exact one-shot variance of all methods.

* Use the argument `-g` to specify distances $\gamma$ to be used in the experiments, for example: `-g 0.5 0.1 0.001` 

#### General Arguments

For the first two scripts (that run all the methods that were used in the experiments of the thesis):

* Use the `--no-fd`, `--no-bd`,`--no-cd`,`--no-spsa`,`--no-ps`, `--no-gps` flags if you want to not compute the gradient with a certain method

* Use `--fd_h`, `--bd_h`, `--cd_h`, `--gps_gamma`, `--spsa_h`,`--spsa_count` to specify hyperparameters for the methods

For all four experiment scripts:

* Use `--seed` to specify the numpy random seed

* Use `--shots_start`, `--shots_stop`, `--shots_step` to specify the list of shotcounts per measurement to use for the approximations. In `gradient_sample_shotslist.py` big shotcounts can rapidly increase the runtime! 

* Use `--ansatz` to specify the ansatz. Possible values: `ESU2` (Qiskits EfficientSU2 Ansatz), `CRE` (Cross Resonance + Euler Rotations Ansatz), `LHE` (Layered Hardware Efficient Ansatz)

* Use `--observable` to specify the observable. Possible values: `Diagonal` (Diagonal Observable with values $2i$ on main diagonal), `LiH` (LiH VQE Observable), `HeH` (HeH VQE Observable), `H2` (H2 VQE Observable)

* Use `--metric` to specify the metric used to compute the gradient errors. Possible values: `MSE` (Mean squared error), `MAE` (Mean absolute error)


### How to use:

Run:

`poetry run .\src\scripts\<scriptname>`

for the script of your choice, starting in the main folder.

The scripts will generate plots in the `plots` directory. 
Also you can look at all the plots and parameters of the experiments in the mlflow UI.

Run:

`poetry run mlflow ui`

Then open the displayed link (normally `http://127.0.0.1:5000`) in your browser.


