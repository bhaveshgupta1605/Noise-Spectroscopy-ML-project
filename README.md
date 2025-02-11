# Training Data Generation Functions for Qubit Decoherence Analysis

This script (`train_data_generation_functions.py`) provides a set of functions for generating training data to be used in machine learning models aimed at understanding and mitigating qubit noise. The core idea is to simulate noise spectra and then calculate the corresponding coherence decay curves, effectively creating a synthetic dataset. This dataset can then be used to train models to predict noise characteristics from observed decoherence.

**Based on:**

*   B. Gupta, V. Joshi, U. Kandpal, P. Mandayam, N. Gheeraert, S. Dhomkar, "Expedited Noise Spectroscopy of Transmon Qubits", arXiv.2502.00679 (2025)
*   <https://doi.org/10.48550/arXiv.2502.00679>
*   D.F. Wise, J.J.L. Morton, and S. Dhomkar, “Using deep learning to understand and mitigate the qubit noise environment”, PRX Quantum 2, 010316 (2021) 
*   <https://doi.org/10.1103/PRXQuantum.2.010316>
*   F. Meneses, D. F. Wise, D. Pagliero, P. R. Zangara, S. Dhomkar, C. A. Meriles, "Toward Deep-Learning-Assisted Spectrally Resolved Imaging of Magnetic Noise", Physical Review Applied, 18 024004 (2022)
*   <http://dx.doi.org/10.1103/PhysRevApplied.18.024004>


## Overview

The script implements functions to:

1.  **Generate Filter Functions:**  Calculates filter functions corresponding to specific dynamical decoupling pulse sequences like CPMG.
2.  **Generate Noise Spectra:** Simulates various qubit noise spectra, including $1/f$-like and Lorentzian-like spectra, which are common in superconducting qubits.
3.  **Calculate Coherence Curves:** Evaluates the coherence decay curve corresponding to a given noise spectrum using the filter function.
4.  **Add Noise:** Augments the data by adding random Gaussian noise to the simulated coherence curves.
5.  **Prepare Training Data:**  Provides functions to interpolate and format the generated data into a suitable format for training machine learning models.
6.  **Visualize Data:** Includes functions to plot the distribution of T2 and stretching parameters.

## Dependencies

The script requires the following Python libraries:

*   `numpy`: For numerical computations and array manipulation.
*   `scipy.interpolate`: Specifically, `interp1d` for interpolating data.
*   `scipy.optimize`: Specifically, the `curve_fit` function for fitting curves to data (this dependency comes from `expt_data_analysis.py`).
*   `tqdm.notebook`: For displaying progress bars during iterative processes in Jupyter notebooks.
*   `plotly`: For creating interactive plots.
*   `random`: For generating random numbers.
*   `expt_data_analysis`: A custom module (assumed to be in the same directory) containing functions for fitting experimental data. **Important:** This script relies on `expt_data_analysis.py`, which is not included but must be available.

You can install the necessary libraries using pip:
pip install numpy scipy tqdm plotly


You'll also need to ensure `expt_data_analysis.py` is in the same directory or that Python can find it on its path.

## Usage

The general workflow is:

1.  **Define Parameters:**  Set parameters for the noise spectra, filter functions, and training data generation process. Key parameters include the T2 range, frequency range, number of pulses in the CPMG sequence, pi-pulse length, time vector, and noise level.
2.  **Generate Smooth Data:** Use `SmoothTrainData` to generate a set of smooth coherence curves and corresponding noise spectra without added noise.
3.  **Add Noise:**  Use functions like `prepare_trainData` or `NoisyTrainData` to add random noise to the smooth coherence curves.
4.  **Prepare for Training:** Format the data for input into a machine learning model.  This may involve interpolation, normalization, and creating appropriate input and output arrays.

### Key Functions

Here's a breakdown of the key functions in the script:

*   **`cpmgFilter(n: int, Tmax: float) -> np.ndarray`**: Generates an array of time points for a CPMG filter.  `n` is the number of pulses, and `Tmax` is the maximum time.
*   **`getFilter(n: int, w0: float, piLength: float, Tmax: float) -> np.ndarray`**: Returns the filter function for a given CPMG pulse sequence.  `w0` is the angular frequency, and `piLength` is the length of the pi pulse.
*   **`transmon_noise(T2: np.ndarray, w: np.ndarray) -> np.ndarray`**: Calculates the transmon noise spectrum based on the given T2 values and frequency vector.  This function simulates a noise spectrum with $1/f^\alpha$ characteristics.
*   **`moving_average(x: np.ndarray, width: int) -> np.ndarray`**: Calculates the moving average of a 1D array, used for smoothing the noise spectrum.
*   **`getCoherence(S: np.ndarray, w0: np.ndarray, T0: np.ndarray, n: int, piLength: int) -> np.ndarray`**: Calculates the coherence decay curve corresponding to a given noise spectrum `S`.
*   **`SmoothTrainData(T_in: np.ndarray, T2: np.ndarray,w: np.ndarray, n_pulse: int, piLength: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]`**:  Generates smooth training data (coherence curves and noise spectra) without added noise.  It also calls `get_fitPar` (from `expt_data_analysis.py`) to fit the generated coherence curves and extract T2 and stretching parameters.
*   **`T2_p_distribution(fit_par: tuple[np.ndarray, np.ndarray]) -> tuple[go.Figure, go.Figure]`**:  Generates histograms of the T2 and stretching parameter distributions, allowing for visual inspection of the generated data.
*   **`interpData(x: np.ndarray, y: np.ndarray, xNew: np.ndarray) -> np.ndarray`**: Interpolates data using a linear interpolation.
*   **`interpolte_c_expt_data(expt_T2_data: np.ndarray, expt_t_data: np.ndarray, T_train: np.ndarray) -> np.ndarray`**: Interpolates experimental data to generate `c_exp` values for given `T_train` values.  This allows you to use experimental data as part of your training set.
*   **`prepare_trainData(c_in: np.ndarray, T_in: np.ndarray, T_train: np.ndarray, noiseMax: float = 0.03) -> np.ndarray`**: Adds random Gaussian noise to the coherence curves.
*   **`prepare_expData(c_in: np.ndarray, T_in: np.ndarray, T_train: np.ndarray, cutOff: float = 0.03) -> np.ndarray`**: Processes experimental data by interpolating it to the training time points and applying a cutoff.
*   **`NoisyTrainData(c_in: np.ndarray, s_in: np.ndarray, w_in: np.ndarray, T_in: np.ndarray, T_train: np.ndarray, w_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]`**: Combines the steps of interpolation and noise addition to generate the final noisy training data.

### Example

Here's an example of how to use the `SmoothTrainData` function:

mport numpy as np
from train_data_generation_functions import SmoothTrainData
Define parameters
T_in = np.geomspace(1.8e-6, 750.05e-6, 451) # Time vector
T2 = np.linspace(100e-6, 370e-6, 15001) # T2 values
w = np.flipud(np.logspace(2.0, 10.0, 16001)) # Frequency vector
n_pulse = 32 # Number of pulses
piLength = 48e-9 # Pi-pulse length
Generate smooth training data
c_in, s_in, w_in, fit_par = SmoothTrainData(T_in, T2, w, n_pulse, piLength)
Print the shapes of the output arrays
print("Shape of c_in:", c_in.shape)
print("Shape of s_in:", s_in.shape)
print("Shape of w_in:", w_in.shape)
print("Shape of fit_par:", fit_par.shape) # T2 values
print("Shape of fit_par1:", fit_par1.shape) # Stretching parameters


### Figure Template

The script uses a predefined `plotly` figure template (`fig_template`) to ensure consistent styling of the plots. You can customize this template to modify the appearance of the plots.  The template includes settings for:

*   Plot size (width and height)
*   Axis titles
*   Tick marks
*   Grid lines
*   Font family and size
*   Color palette

## Notes

*   This script is designed to generate synthetic training data for machine learning models. The accuracy of the generated data depends on the validity of the assumptions made about the noise spectra and the qubit system.
*   The `expt_data_analysis.py` file, which is not included, is crucial for this script to work because it contains fitting functions.
*   Careful selection of the parameters (T2 range, frequency range, noise level, etc.) is crucial for generating a realistic and useful training dataset.
*   The `tqdm` progress bars are most effective when running the script in a Jupyter Notebook environment.
*   The `transmon_noise` function includes some randomization.

This script provides a flexible framework for generating training data for qubit decoherence analysis. By adjusting the parameters and functions, you can tailor the generated data to match the characteristics of your specific qubit system and experimental setup.

# Qubit Noise Spectroscopy Network Training Script

This script (`network_functions.py`) trains a 1D convolutional neural network (CNN) to denoise noise spectroscopy data related to qubit characterization. It's designed to learn the relationship between noisy coherence curves and underlying noise spectra, allowing for improved noise estimation in quantum devices.

## Overview

The script performs the following main tasks:

1.  **Model Definition:** Defines a 1D CNN architecture using Keras for denoising noise spectroscopy data.
2.  **Data Loading and Formatting:** Loads training data from an `.npz` file, formats the data for training, and splits it into training and testing sets.  The expected data format is specific to the outputs of `data_generation_functions.py`.
3.  **Training:** Trains the CNN model using the prepared data, employing techniques like learning rate reduction on plateau and callbacks for monitoring performance.
4.  **Evaluation:** Evaluates the trained model on the test set and saves the model, training history, and example predictions.
5.  **Hyperparameter Tuning:** The script is designed to be run with command-line arguments to easily explore different hyperparameter configurations.

**Based on:**

D.F. Wise, J.J.L. Morton, and S. Dhomkar, “Using deep learning to understand and mitigate the qubit noise environment”, PRX Quantum 2, 010316 (2021)

*   <https://doi.org/10.1103/PRXQuantum.2.010316>
*   <http://dx.doi.org/10.1103/PhysRevApplied.18.024004>

## Dependencies

The script requires the following Python libraries:

*   `tensorflow`: For building and training the neural network.
*   `scikit-learn`: For splitting the data into training and testing sets.
*   `numpy`: For numerical computations and array manipulation.
*   `matplotlib`: For plotting training history and test results.
*   `data_generation_functions`: A custom module (assumed to be in the same directory) containing functions for preparing the training data.  The training data needs to be formatted according to the expectations of `generate_final_data()`.

You can install these dependencies using pip:


You'll also need to ensure `data_generation_functions.py` is in the same directory or that Python can find it on its path.

## Usage

### 1. Data Preparation

The script expects training data to be stored in an `.npz` file.  This `.npz` file should be created by a script like `data_generation_functions.py` and should contain the following arrays:

*   `c_in`: Input coherence data (noisy).
*   `T_in`: Time vector corresponding to the input coherence data.
*   `s_in`:  Target noise spectra.
*   `w_in`:  Frequency vector corresponding to the noise spectra.
*   `T_train`: Time vector for training data (based on the experimental data).
*   `w_train`: Frequency vector for training data.

The script loads the data using `np.load("data/Mar6_x32_noisy.npz")`. **Modify this path** to point to your data file. Place the data file in a directory named `data/` or adjust the path accordingly.

### 2. Running the Script

The script is designed to be run from the command line, allowing for easy adjustment of hyperparameters.  Here's an example: python network_functions.py --batch_size 64 --epochs 100 --filters 16 --kernel_size 21 --initial_lr 0.001 --min_lr 0.00001 --patience 5 --min_delta 0.1 --verbose 1 --net_type 1


You can create a bash script (as provided at the beginning of the file) to automate running the script with different hyperparameter settings.

### 3. Command-Line Arguments

The script accepts the following command-line arguments:

*   `--batch_size`: Batch size for training. Default: 64
*   `--epochs`: Number of training epochs. Default: 20
*   `--filters`: Number of filters in the convolutional layers. Default: 40
*   `--kernel_size`: Size of the convolutional kernel. Default: 5
*   `--initial_lr`: Initial learning rate. Default: 0.001
*   `--min_lr`: Minimum learning rate for learning rate reduction. Default: 1e-6
*   `--min_delta`: Minimum change in loss to qualify as an improvement for learning rate reduction. Default: 0.5
*   `--patience`: Number of epochs with no improvement before reducing the learning rate. Default: 6
*   `--verbose`: Verbosity level (0 or 1). Default: True
*   `--net_type`:  An integer representing the type of network (potentially for different architectures).  Currently unused in the provided code. Default: 1

### 4. Output

The script generates the following output:

*   **Trained Model:** Saves the trained Keras model to the `TRAINED_NETWORKS/` directory with a filename that includes the hyperparameters used for training (e.g., `TRAINED_NETWORKS/MODEL_... .h5`).
*   **Training History Plot:**  Saves a plot of the training and validation loss (Mean Absolute Percentage Error - MAPE) over epochs to `TRAINED_NETWORKS/VAL_ACC_HISTORY_... .pdf`.
*   **Test Results Plot:** Saves a plot comparing the predicted noise spectra to the true noise spectra for a random subset of the test data to `TRAINED_NETWORKS/MODEL_TEST_... .pdf`.
*   **Console Output:** Prints the hyperparameters used for training, the training time, and the final validation loss (accuracy).

## Function Details

*   **`get_model(filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size)`**: This function defines the 1D CNN architecture. It consists of multiple convolutional layers with ReLU activation, max pooling layers, upsampling layers, and a final dense layer to output the predicted noise spectrum.  The architecture includes:
    *   Input layer: Takes the coherence curve as input.
    *   Convolutional layers: Extract features from the input data.
    *   Max pooling layers: Reduce the dimensionality of the data and capture important features.
    *   Up-sampling layers: Increase the dimensionality of the data to match the size of the output.
    *   Dropout layer: Prevents overfitting by randomly dropping out neurons during training.
    *   Dense layer: Maps the learned features to the output noise spectrum.

*   **`generate_final_data( c_data, T_in, s_data, w0, T_train, w_train )`**: This function, which is assumed to be in `data_generation_functions.py`, prepares the data for training. It likely involves interpolating the data to a common time axis and potentially adding noise.  The exact implementation is not visible in the provided code.

## Notes

*   The performance of the trained model depends heavily on the quality and quantity of the training data. It's crucial to generate a realistic and diverse training dataset that accurately represents the noise environment of the qubit system.
*   The choice of hyperparameters (e.g., number of filters, kernel size, learning rate) can significantly impact the model's performance. Experimentation with different hyperparameter settings is recommended to find the optimal configuration for your specific application.
*   The script uses Mean Absolute Percentage Error (MAPE) as the loss function. You may want to explore other loss functions depending on the characteristics of your data and the specific goals of your analysis.  MAPE is sensitive to small values in the target variable.
*   The script uses `tensorflow.keras.optimizers.legacy.Adam`. Depending on your TensorFlow version, you might need to use `tensorflow.keras.optimizers.Adam` directly.
*   The script saves the model and plots to the `TRAINED_NETWORKS/` directory. Make sure this directory exists before running the script.

This script provides a starting point for training a neural network to denoise qubit noise spectroscopy data. By customizing the model architecture, training parameters, and data preprocessing steps, you can adapt the script to meet the specific requirements of your application.

# Experimental Data Analysis Script

This script (`expt_data_analysis.py`) is designed for analyzing experimental data related to qubit coherence measurements, specifically T1 and T2 decay. It includes functions for data normalization, curve fitting using exponential and stretched-exponential models, and generating interactive plots to visualize the results.

## Overview

The script performs the following main tasks:

1.  **Data Loading and Preprocessing:** Loads experimental data from an `.npz` file, normalizes the T2 data, and prepares the data for further analysis.
2.  **Curve Fitting:** Fits the T1 and T2 data to appropriate exponential functions (simple exponential for T1 and stretched exponential for T2) to extract relevant parameters.
3.  **Visualization:** Generates interactive plots of the coherence curves along with the fitted curves, allowing for visual inspection of the data and fit quality.

## Dependencies

The script requires the following Python libraries:

*   `numpy`: For numerical computations and array manipulation.
*   `scipy.optimize`: Specifically, the `curve_fit` function for fitting curves to data.
*   `tqdm.notebook`: For displaying progress bars during iterative processes (e.g., fitting multiple curves).  This is designed for Jupyter notebooks.
*   `plotly`: For creating interactive plots.

You can install these dependencies using pip:
Example: 
pip install numpy scipy tqdm plotly

## Usage

### 1. Data Preparation

The script expects the experimental data to be stored in an `.npz` file.  The `.npz` file should contain the following arrays:

*   `xval`:  The time values at which the measurements were taken (in microseconds). This should be a 1D numpy array.
*   `pop_t1`:  The population of the `|1>` qubit state as a function of time. This should be a 2D numpy array, where each row corresponds to a different qubit.
*   `pop_t2_[protocol]`: The population of the `|0>` qubit state as a function of time for a specific dynamical decoupling protocol (e.g., Hahn echo, X32, XY4).  Replace `[protocol]` with the actual protocol name used in your experiment.  This should be a 2D numpy array, where each row corresponds to a different qubit.

Example: 
expt_data = np.load('your_data_file.npz', allow_pickle=True)
print(expt_data.files)


This will print the names of the arrays stored in the `.npz` file, allowing you to verify that the required data is present.

### 2. Running the Script

1.  **Modify the `main()` function:**  You will need to modify the `main()` function in the script to point to the correct path for your `.npz` data file.

    ```
    def main():
        # Load experimental data
        expt_data = np.load('path/to/your/data.npz', allow_pickle=True)
        print(expt_data.files)
    ```

2.  **Choose the T2 data:** Select the appropriate `pop_t2_[protocol]` data to be used for T2 analysis, based on the dynamical decoupling protocol you want to analyze.

    ```
    expt_T2_data = expt_data['pop_t2_X32']  # Change 'pop_t2_X32' to the correct key
    ```

3.  **Run the script:** Execute the script from the command line:

    ```
    python expt_data_analysis.py
    ```

    Alternatively, you can run the script in a Jupyter Notebook.

### 3. Output

The script will:

*   Print the keys of the loaded `.npz` file to the console.
*   Print the fitted T2 values (in microseconds) and the stretching parameter 'p' to the console.
*   Display an interactive plot of the coherence curves for each qubit, along with the corresponding fitted curves. The legend of each fitted curve displays the T2 value (in microseconds) and the p value of the fit.

## Function Details

Here's a breakdown of the key functions in the script:

*   **`normalise(data: float) -> float`**: Normalizes the input data to a range between -1 and 1.

*   **`normT2data(expt_T2_data: np.ndarray) -> np.ndarray`**: Normalizes the T2 data by dividing each row by the mean of the first 3 elements of that row.  This helps to account for variations in the initial signal amplitude.

*   **`simpleExp(T0: float, T1: float) -> float`**: Defines a simple exponential decay function.  Used for fitting T1 data.

*   **`fit_simpleExp(expt_T1_data: np.ndarray, T0: float, bounds: tuple) -> tuple[float, float]`**: Fits the `simpleExp` function to the T1 data and returns the fitted T1 value and its error.

*   **`stretchExp(T0: float, T2: float, p: float, A: float) -> float`**: Defines a stretched exponential decay function. Used for fitting T2 data.

*   **`fit_stretchExp(C: float, T0: float, bounds: tuple) -> tuple[float, float, float, float, float, float]`**: Fits the `stretchExp` function to the T2 data and returns the fitted T2 value, stretching parameter `p`, amplitude `A`, and their respective errors.

*   **`c_expt_data(norm_T2_data: np.ndarray, expt_T1_data: np.ndarray) -> np.ndarray`**: Calculates the experimental coherence data by combining the normalized T2 data and the T1 data.

*   **`get_c_expt_fitPar_fitCurves(expt_c_data: np.ndarray, expt_t_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]`**: Fits the stretched exponential function to the experimental coherence data and returns the fit parameters (T2, p, A) and the fitted curves.

*   **`get_fitPar(c_check: np.ndarray, T_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]`**:  An alternative implementation for fitting multiple coherence curves, useful for training data generation or checking T2 and p distributions.

*   **`get_c_expt_plot(expt_c_data: np.ndarray, expt_t_data: np.ndarray) -> go.Figure`**: Generates an interactive plot of the experimental coherence data and the fitted curves using `plotly`.

*   **`heatmap(expt_t_data: np.ndarray, expt_c_data: np.ndarray) -> go.Figure`**: Creates a heatmap visualization of the coherence data.

## Figure Template

The script uses a predefined `plotly` figure template (`fig_template`) to ensure consistent styling of the plots.  This template defines the plot size, axis titles, font, and color scheme. You can modify the template to customize the appearance of the plots.

## Notes

*   The script assumes that the time data (`xval`) is in microseconds.
*   The bounds for the curve fitting parameters (T2, p, A) in `fit_stretchExp` may need to be adjusted depending on the characteristics of your experimental data.  Carefully choose the bounds to ensure reliable fitting results.
*   The `tqdm` progress bar is most effective when running the script in a Jupyter Notebook environment.
*   The script saves the generated plots in the pdf format. Specify the path in `pio.write_image(fig,path,format='pdf')` to save the plots.
