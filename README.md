# Noise Spectroscopy using ML

## To evaluate a decoherence curve from an underlying noise spectrum
Siddharth Dhomkar, Department of Physics, IIT Madras

**Based on**:

D.F. Wise, J.J.L. Morton, and S. Dhomkar, “Using deep learning to understand and mitigate the qubit noise environment”, PRX Quantum 2, 010316 (2021)

*   https://doi.org/10.1103/PRXQuantum.2.010316
*   http://dx.doi.org/10.1103/PhysRevApplied.18.024004

## Functions to generate coherence curve from a given set of noise spectra

It is virtually impossible to extract the underlying noise spectrum from a decoherence curve. However, it is possible to obtain an unique coherence decay, if the precise functional form of the noise spectrum is known. This observation is utilized to generate the training datasets.

Qubit noise spectra, $S(\omega)$, can be of various functional forms including $1/f$-like, Lorentzian-like, double-Lorentzian-like etc. It is possible to simulate such noise spectra and corresponding decoherence curve, $C(t)$, can then be evaluated consequently using following relation:

$$- ln C(t) =  \int_{0}^{\infty}\frac{d\omega}{\pi}S(\omega)\frac{F(\omega t)}{\omega^{2}}$$

where, $F(\omega t)$ is the filter function associated with the decoupling protocol used for probing the decoherence dynamics.

(The detailed explanation of data generation process can be found in the manuscript.)

**Generate the filter function**

References pertaining to superconducting qubit noise spectra:

*   https://doi.org/10.1038/ncomms1856
*   https://www.nature.com/articles/ncomms12964
*   https://doi.org/10.1103/PhysRevApplied.18.044026
*   https://doi.org/10.1103/PhysRevResearch.3.013045
*   http://dx.doi.org/10.1103/PhysRevB.79.094520
*   http://dx.doi.org/10.1103/PhysRevB.77.174509
*   https://doi.org/10.1063/1.5089550

## Experimental data analysis

Imports the necessary libraries for data manipulation, curve fitting, progress tracking, and interactive plotting.
- numpy: Fundamental package for scientific computing with Python.
- scipy.optimize.curve_fit: Module for fitting a function to data.
- tqdm.notebook.trange: Provides a progress bar for loops in Jupyter notebooks.
- plotly.graph_objs as go: Interface to the `plotly` library for creating interactive plots.

### Defines the layout settings for the figure template.
Layout settings:
- Template style: 'simple_white+presentation'
- Autosize: False
- Width: 800
- Height: 600
- X-axis settings: Title, ticks, mirror, linewidth, tickwidth, ticklen, showline, showgrid, zerolinecolor
- Y-axis settings: Title, ticks, mirror, linewidth, tickwidth, ticklen, showline, showgrid, zerolinecolor
- Font family: mathjax
- Font size: 16
- Colorway: ["#d9ed92","#b5e48c","#99d98c","#76c893","#52b69a","#34a0a4","#168aad","#1a759f","#1e6091","#184e77"]

### Load the experimental data as npz file from the specified path: 
expt_data = np.load(path,allow_pickle=True)

expt_data.files ---> should contain 'pop_t1','pop_t2' and 'xval' data files: 
print(f'files in expt_data: {expt_data.files}')
files in expt_data:
- 'xval': Experiemnt time array
- 'pop_t1': population of |1> at corresponding time array
- 'pop_t2_[hahn/X32/XY4/...]': population of |0> at corresponding time array
Then assign the data to the variables:
- expt_t_data = expt_data['xval'] ----> shape = (size of time array,)
- expt_T1_data = expt_data['pop_t1']--> shape = (indexing of qubit number, pop_t1 data value at that time)
- expt_T2_data = expt_data['pop_t2']--> shape = (indexing of qubit number, pop_t2 data value at that time)

### Normalise the input data by subtracting 0.5 and then dividing by 0.5. 
Parameters:
- data (float) : The input data to be normalized.
Returns:
- float : The normalized data.

### Normalize the given T2 data array by dividing each row by the mean of the first 3 elements of that row.
Parameters:
- expt_T2_data (numpy.ndarray): The experimental T2 data array to be normalize
  
Returns:
- (numpy.ndarray): the normalized T2 data array

### A function that calculates the exponential decay of T0 divided by T1.

Parameters:
- T0: The base value used in the exponential calculation.
- T1: The divisor value used in the exponential calculation.
  
Returns:
- The result of the exponential calculation.

### Fit a simple exponential function to the given T1 data using scipy curve fit method.

Parameters:
- expt_T1_data (np.ndarray): The experimental T1 data to fit the simple exponential function to.
- T0 (float): The initial guess for the fitting.
- bounds (tuple): The bounds for the fitting parameter T0.

Example:
- T0 = 10e-6
- bounds = ([10e-6],[1000e-6])
  
Returns:
- T1: The fitted T1 value.
- T1err: The error in the fitted T1 value.

### A function that calculate Stretched-exponential function, based on the input parameters T0, T2, p, and A.

Parameters:
- T0 (float): The base value used in the exponential calculation.
- T2 (float): The divisor value used in the exponential calculation.
- p (float): The streaching parameter.
- A (float): The amplitude parameter.

Returns: 
- C (float): The value of Stretched-exponential function.

### Fit a stretch exponential curve to the given data.

Parameters:
- C (float): The data to fit the curve to using scipy curve fit method.
- T0 (float): The independent variable for the curve fit.
- bounds (tuple): The bounds for the curve fit. Example: bounds = ([100e-6,1,0.97],[400e-6,2,1.03])

Returns:
- Tuple: The fitted parameters T2, p, A, and their respective errors T2err, perr, Aerr.

### Calculate experimental coherence curves data based on normalized T2 and experimental T1 data.

Parameters:
- norm_T2_data (np.ndarray): Array of normalized T2 data.
- expt_T1_data (np.ndarray): Array of experimental T1 data.

Returns:
- (np.ndarray): Array of experimental coherence curves data.

### Generate the exponential fit parameters and fit curves for the given experimental data.

Parameters:
- expt_c_data (np.ndarray): The experimental concentration data.
- expt_t_data (np.ndarray): The experimental time data.

Returns:
- tuple[np.ndarray, np.ndarray]: A tuple containing the exponential fit parameters and fit curves.

### Generate the T2 values and corresponding parameters for each row in the input array using the fit_stretchExp function.

Parameters:
- c_check (numpy.ndarray): Input array of shape (n, m) where n is the number of rows and m is the number of columns.
- T_train (numpy.ndarray): Training data array.

Returns:
- Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing two numpy arrays. The first array contains the T2 values multiplied by 1e6 and rounded to 1 decimal place. The second array contains the corresponding parameters rounded to 2 decimal places.

### Generate a plot of experimental coherence data against evolution time for each qubit. 

Parameters:
- expt_c_data (numpy.ndarray): Array of coherence data for each qubit.
- expt_t_data (numpy.ndarray): Array of evolution time data.
- 
Returns:
- go.Figure: A plot displaying the coherence data for each qubit and corresponding fit curves.

### Create a heatmap plot based on the experimental transmon data and control data.

Parameters:
- expt_t_data: The experimental transmon data
- expt_c_data: The experimental control data

Returns:
- fig: The generated heatmap plot
