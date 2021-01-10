# ESCSPKF toolbox (Python version)

Python version of Gregory Plett's ESCSPKF toolbox with a sigma-point Kalman filter (SPKF) cell SOC estimator. The original code is written in Matlab which is available in the ESCSPKF toolbox at
[mocha-java.uccs.edu/BMS2/](http://mocha-java.uccs.edu/BMS2/).

## SPKF cell SOC estimator

The SPKF cell SOC estimator is the `SPKF.py` file, it loads `plett_blend_np1_hys1_modeldyn.json`, a 1RC Enhanced-self Correcting (ESC) model generated with the Gavin Wiggins Python version of Gregory Plett's enhanced self-correcting (ESC) model, available at [github.com/batterysim/esctoolbox-python](https://github.com/batterysim/esctoolbox-python). The `SPKF.py` estimator also imports dynamic tests data files to test the 1RC ESC model and the `funcs.py` functions as modules.

See the comments in each file for more information.

## EKF cell SOC estimator

Pending

## Installation

Requires Python 3.6, Matplotlib, NumPy, and Pandas. The preferred method to
install Python 3 and associated packages is with the Anaconda or Miniconda
distribution available at
[continuum.io/downloads](https://www.continuum.io/downloads).

## Usage

Clone or download the files to your local machine. Start iPython and run the `SPKF.py` file.