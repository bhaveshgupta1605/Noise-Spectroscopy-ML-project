# Qubit Noise Spectroscopy with Deep Learning

This repository contains Python code for using deep learning to analyze and understand noise in qubit systems.  It's based on the work described in:

> D.F. Wise, J.J.L. Morton, and S. Dhomkar, “Using deep learning to understand and mitigate the qubit noise environment”, PRX Quantum 2, 010316 (2021)

## Overview

The code allows you to:

1.  **Generate training data:**  Simulate noise spectra and corresponding coherence decay curves.
2.  **Train a neural network:**  Learn the relationship between coherence curves and noise spectra.
3.  **Analyze experimental data:**  Use the trained network to predict the noise spectrum from experimental coherence measurements.
4.  **Optimal Dynamic Decoupling Sequence Generation:** Generate optimized pulse sequence to maximize coherence.

## Repository Structure

The repository is organized as follows:

*   `MAIN.py`: The main script for training and evaluating the neural network.
*   `optimalDD_functions.py`: Contains functions related to generating and analyzing optimal dynamic decoupling (DD) sequences.
*   `expt_data_analysis.py`:  Functions for loading, normalizing, and analyzing experimental data, including fitting coherence curves.
*   `train_data_generation_functions.py`:  Functions for generating synthetic training data, including noise spectra and coherence curves.
*   `data/`: (This directory is assumed) Contains the `Mar6_x32_noisy.npz` dataset (or similar).
*   `TRAINED_NETWORKS/`: (This directory is assumed) Where trained network models are saved.

## Dependencies

*   tensorflow
*   scikit-learn
*   numpy
*   matplotlib
*   scipy
*   plotly

You can install the necessary packages using `pip`:

pip install tensorflow scikit-learn numpy matplotlib scipy plotly

## Data Format

The code uses `.npz` files for data storage.  Specifically, the `MAIN.py` script expects a file named `"data/Mar6_x32_noisy.npz"` (the path is hardcoded). This file should contain the following arrays:

*   `c_in`:  Coherence data.
*   `T_in`:  Time vector for data generation.
*   `s_in`:  Noise spectra data.
*   `w0`:  Omega (frequency) vector for data generation.
*   `T_train`:  Time vector for training data (based on the experimental data).
*   `w_train`:  Omega vector for training data.

## Usage

### 1. Data Generation (using `train_data_generation_functions.py`)

The `train_data_generation_functions.py` script provides functions to simulate noise spectra and their corresponding coherence curves.

*   **`cpmgFilter(n, Tmax)`:** Generates an array of time points for a CPMG pulse sequence.

    *   `n`: Number of π pulses.
    *   `Tmax`: Total duration of the sequence.

*   **`getFilter(n, w0, piLength, Tmax)`:**  Calculates the filter function for a CPMG sequence.

    *   `n`: Number of π pulses.
    *   `w0`: Angular frequency.
    *   `piLength`: Duration of the π pulse.
    *   `Tmax`: Total duration of the sequence.

*   **`transmon_noise(T2, w)`:** Generates a transmon noise spectrum based on a T2\* value and frequency vector.

    *   `T2`: T2\* values.
    *   `w`: Frequency vector.

*   **`getCoherence(S, w0, T0, n, piLength)`:**  Calculates the coherence decay curve given a noise spectrum, filter function, and time vector.

    *   `S`: Noise spectrum.
    *   `w0`: Frequency vector.
    *   `T0`: Time vector.
    *   `n`: Number of π pulses.
    *   `piLength`: Duration of the π pulse.

*   **`SmoothTrainData(T_in, T2, w, n_pulse, piLength)`:** Generates smooth training data consisting of coherence curves, noise spectra, and fitting parameters.

    *   `T_in`: Time vector.
    *   `T2`: T2\* values.
    *   `w`: Frequency vector.
    *   `n_pulse`: Number of π pulses.
    *   `piLength`: Duration of the π pulse.

*   **`NoisyTrainData(c_in, s_in, w_in, T_in, T_train, w_train)`:** Adds random noise to the generated coherence curves to create a more realistic training dataset.

    *   `c_in`: Input c values.
    *   `s_in`: Input s values.
    *   `w_in`: Input w value.
    *   `T_in`: Input T values.
    *   `T_train`: Training T values.
    *   `w_train`: Training w values.
    *   `prepare_trainData`: Adding noise to the data using `np.random.normal`.
    *   `interpData`: Interpolating the data using `scipy.interpolate.interp1d`.

**Example Usage (Illustrative):**

import numpy as np
from train_data_generation_functions import transmon_noise, getCoherence
Define parameters
T2_values = np.linspace(10e-6, 100e-6, 10) # Example T2 values
w_values = np.logspace(3, 8, 1000) # Example frequency values
T_values = np.geomspace(1e-6, 100e-6, 50) # Example time values
n_pulses = 32
pi_length = 48e-9
Generate noise spectra
s_values, w_out = transmon_noise(T2_values, w_values)
Generate coherence curves
c_values = getCoherence(s_values, w_out, T_values, n_pulses, pi_length)
Now you can save c_values, s_values, w_out and T_values to a .npz file for training
e.g., np.savez("my_training_data.npz", c_in=c_values, s_in=s_values, w0=w_out, T_in=T_values)


### 2. Network Training (using `MAIN.py`)

The `MAIN.py` script trains a neural network to predict the noise spectrum from a given coherence curve.

**Command-line arguments:**

The script accepts several command-line arguments to configure the network architecture and training process:

*   `--batch_size`: Batch size for training (default: 64).
*   `--epochs`: Number of training epochs (default: 20).
*   `--filters`: Number of filters in the convolutional layers (default: 40).
*   `--kernel_size`: Kernel size in the convolutional layers (default: 5).
*   `--initial_lr`: Initial learning rate (default: 1e-3).
*   `--min_lr`: Minimum learning rate (default: 1e-6).
*   `--min_delta`: Minimum change in loss to qualify as an improvement (default: 0.5).
*   `--patience`: Number of epochs with no improvement after which learning rate will be reduced (default: 6).
*   `--nb`:  (Purpose unclear from code, likely a network ID or similar).
*   `--verbose`: Verbosity mode (0 or 1) (default: True).
*   `--net_type`:  Type of network architecture to use (default: 1).

**Example Usage:**

python MAIN.py --batch_size 128 --epochs 50 --filters 64 --initial_lr 1e-4


This command will train the network using a batch size of 128, for 50 epochs, with 64 filters, and an initial learning rate of 1e-4.

**Key steps in `MAIN.py`:**

1.  **Load data:** Loads the training data from `"data/Mar6_x32_noisy.npz"`.
2.  **Data formatting:** Calls `generate_final_data` (likely from a helper module) to format the data for training.
3.  **Model creation:** Calls `get_model` (likely from a helper module, not provided) to create the neural network model.  The architecture depends on `net_type`.
4.  **Compilation:** Compiles the model with the Adam optimizer and MAPE (Mean Absolute Percentage Error) loss function.
5.  **Training:** Trains the model using `model.fit`, with a `ReduceLROnPlateau` callback to adjust the learning rate during training.
6.  **Saving:** Saves the trained model to the `TRAINED_NETWORKS/` directory.
7.  **Testing:**  Makes predictions on the test set and generates plots comparing predicted and actual noise spectra.

### 3. Experimental Data Analysis (using `expt_data_analysis.py`)

The `expt_data_analysis.py` script provides functions to load, normalize, and analyze experimental data.

*   **`normalise(data)`:** Normalizes experimental data.
*   **`normT2data(expt_T2_data)`:** Normalizes T2 data.
*   **`simpleExp(T0, T1)`:** Calculates a simple exponential decay.
*   **`fit_simpleExp(expt_T1_data, T0, bounds)`:** Fits a simple exponential function to T1 data.
*   **`stretchExp(T0, T2, p, A)`:** Calculates a stretched exponential function.
*   **`fit_stretchExp(C, T0, bounds)`:** Fits a stretched exponential function to data.
*   **`c_expt_data(norm_T2_data, expt_T1_data)`:** Calculates experimental coherence curves.
*   **`get_c_expt_fitPar_fitCurves(expt_c_data, expt_t_data)`:** Fits stretched exponentials to experimental coherence curves and extracts the fitting parameters.
*   **`get_fitPar(c_check, T_train)`:** Fits multiple coherence curves to obtain values of T2 and p (stretching factor).
*   **`get_c_expt_plot(expt_c_data, expt_t_data)`:** Generates a plot of experimental coherence data and their fits.
*   **`heatmap(expt_t_data, expt_c_data)`:** Creates a heatmap visualization of the experimental data.

**Example Usage:**

import numpy as np
from expt_data_analysis import normT2data, c_expt_data, get_c_expt_plot
Load experimental data
expt_data = np.load('data/T1_T2_data.npz', allow_pickle=True) # Replace with your data path
expt_t_data = expt_data['xval']
expt_T1_data = expt_data['pop_t1']
expt_T2_data = expt_data['pop_t2_X32']
Normalize the data
norm_X32_data = normT2data(expt_T2_data)
expt_c_data_X32 = c_expt_data(norm_X32_data, expt_T1_data)
Plot the coherence curves
get_c_expt_plot(expt_c_data_X32, expt_t_data)


### 4. Optimal Dynamic Decoupling (using `optimalDD_functions.py`)

The `optimalDD_functions.py` script provides functions to design optimal dynamic decoupling (DD) sequences to minimize the effects of noise.

*   **`cpmgFilter(n, Tmax)`:** Generates CPMG pulse timings.
*   **`getFilter(n, w0, piLength, Tmax)`:** Calculates the filter function for a CPMG sequence.
*   **`arbFilter(w0, piLength, tpi, Tmax)`:** Calculates the filter function for an *arbitrary* pulse sequence.
*   **`getCoherence(S, w0, T0, n, piLength)`:** Calculates coherence under a CPMG sequence.
*   **`optCoherence(S, w0, piLength, tpi, Tmax)`:** Calculates coherence under an *arbitrary* pulse sequence.
*   **`stretchExp(T0, T2, p, A)`:** Stretched exponential function.
*   **`fit_stretchExp(C, T0)`:** Fits a stretched exponential.
*   **`predicted_s_plot(w_in, s_in)`:** Plots the predicted noise spectrum.
*   **`w_low_s_low(w_in, w_extra)`:** Generates low-frequency noise data.
*   **`w_extend_s_extend(s_in, w_in, s_low, w_low, s_select)`:** Extends the frequency range of the noise spectrum.
*   **`c_extend(s_extend, w_extend, T_in, piLength)`:** Calculates coherence for the extended spectrum.
*   **`customDD(s_extend, w_extend, piLength, x, Tmax)`:**  Calculates coherence for a custom DD sequence (the function to be minimized in the optimization).
*   **`c_dd_c_cpmg(s_in, w_in, s_low, w_low, data_points, n_plots, n, piLength)`:**  Compares coherence under custom DD and CPMG sequences.
*   **`c_dd_c_cpmg_plot(C_DD, C_CPMG, Tmax, n_plots)`:**  Plots the coherence comparison.
*   **`cpmg_optimal_c_vals(w_low, s_low, w_in, s_in, s_select, n_pi, data_points, piLength)`:** Calculates coherence values for both CPMG and optimized DD sequences.
*   **`comparison_plot(C_vals, C_vals_new, Tmax, n)`:**  Plots a comparison of coherence improvement.

**Optimization Process**

The key idea is to find the pulse timings (`tpi`) that *minimize* the coherence decay. This is done using `scipy.optimize.minimize`. The `customDD` function calculates the *negative* of the coherence, so minimizing it maximizes the coherence.

**Constraints**

The optimization includes linear constraints to ensure that the pulse timings are physically realizable:

*   The pulses must be ordered in time.
*   There must be sufficient time for all pulses within the total sequence duration (`Tmax`).

**Example Usage:**

import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from optimalDD_functions import customDD, cpmgFilter
Example Parameters
n = 8 # Number of pulses
piLength = 48e-9 # Pulse length
Tmax = 300e-6 # Total sequence time
Define objective function (negative coherence)
def objective(x):
# Assuming you have s_extend and w_extend defined elsewhere (noise spectrum data)
# You'll need to load or generate these based on your noise model
global s_extend, w_extend # Access s_extend and w_extend

return customDD(s_extend, w_extend, piLength, x, Tmax) #negative coherence

Initial guess (CPMG timings)
x0 = cpmgFilter(n, Tmax)
Bounds (pulse timings must be greater than half the pulse length)
lower_bounds = np.array([(1/2 + i) * piLength for i in range(n)])
upper_bounds = np.array([Tmax - (n - (2*i + 1)/2) * piLength for i in range(n)]) #upper bound equation
bounds = Bounds(lower_bounds,upper_bounds)
Linear constraints (pulses must be ordered)
coeff_matrix = -np.diag(np.ones(n)) #diagonal matrix
for i in range(n-1):
coeff_matrix[i, i+1] = 1
M = coeff_matrix[:-1] # M matrix
lower_B = piLength*np.ones(n-1) #lower bound condition
upper_B = (Tmax-(n-1)*piLength)*np.ones(n-1) #upper bound condition
linear_constraint = LinearConstraint(M, lower_B, upper_B) #linear constraint in pulse sequence

Optimization
res = minimize(objective, x0, method='SLSQP',
constraints=[linear_constraint],
bounds=bounds) #minimize constraints
print(res) #display result
res.x now contains the optimized pulse timings


**Important:**  The example above is a *skeleton*.  You'll need to:

1.  **Define `s_extend` and `w_extend`:**  These represent your noise spectrum.  You'll likely load these from a file or generate them using the functions in `train_data_generation_functions.py` and  `optimalDD_functions.py` (the `w_extend_s_extend` function, for example).  The noise spectrum is *crucial* for the optimization.
2.  **Adapt the objective function:**  Make sure the `objective` function correctly calls `customDD` with *your* noise spectrum data.

## Notes

*   The code uses hardcoded file paths (e.g., `"data/Mar6_x32_noisy.npz"`).  You should modify these to match your data organization.
*   The `get_model` function is not provided in the given files. You'll need to define this function (likely in a separate module) to create your neural network architecture.
*   The `optimalDD_functions.py` script contains a comment "`# Not complete for last function to compare filter function of cpmg and optimal!!`".  This indicates that the `cpmg_optimal_filter_comp_plot` function is not fully implemented.

## Contributing

Contributions are welcome! Please submit pull requests with bug fixes, improvements, or new features.
