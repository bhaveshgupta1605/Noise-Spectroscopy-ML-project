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



