"""
Imports the necessary libraries for data manipulation, curve fitting, progress tracking, and interactive plotting.
- numpy: Fundamental package for scientific computing with Python.
- scipy.optimize.curve_fit: Module for fitting a function to data.
- tqdm.notebook.trange: Provides a progress bar for loops in Jupyter notebooks.
- plotly.graph_objs as go: Interface to the `plotly` library for creating interactive plots.
"""
import numpy as np
from scipy.optimize import curve_fit
from tqdm.notebook import trange
from plotly import graph_objs as go

# Figure Template
fig_template = go.layout.Template()
"""
Defines the layout settings for the figure template.

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
"""
fig_template.layout = {
    'template': 'simple_white+presentation',
    'autosize': False,
    'width': 800,
    'height': 600,
    # 'opacity': 0.2,
    'xaxis': {
        'title': 'Time (\u03BCs)',
        'ticks': 'inside',
        'mirror': 'ticks',
        'linewidth': 2.5,
        'tickwidth': 2.5,
        'ticklen': 6,
        'showline': True,
        'showgrid': False,
        'zerolinecolor': 'white',
        },
    'yaxis': {
        'title': 'Coherence',
        'ticks': 'inside',
        'mirror': 'ticks',
        'linewidth': 2.5,
        'tickwidth': 2.5,
        'ticklen': 6,
        'showline': True,
        'showgrid': False,
        'zerolinecolor': 'white'
        },
    'font':{'family':'mathjax',
            'size': 16,
            },
    'colorway': ["#d9ed92","#b5e48c","#99d98c","#76c893","#52b69a","#34a0a4","#168aad","#1a759f","#1e6091","#184e77"]
}

"""
  Load the experimental data as npz file from the specified path: 
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
"""

# Normalise experimental decoherence data from 1 to 0
def normalise(data: float) -> float:
  """
  Normalise the input data by subtracting 0.5 and then dividing by 0.5. 

  Parameters:
  - data (float) : The input data to be normalized.

  Returns:
  - float : The normalized data.
  """
  return (data-0.5)/0.5

# Normalisation of T2 signals
def normT2data(expt_T2_data: np.ndarray) -> np.ndarray:
  """
  Normalize the given T2 data array by dividing each row by the mean of the first 3 elements of that row.

  Parameters:
  - expt_T2_data (numpy.ndarray): The experimental T2 data array to be normalized

  Returns:
  - (numpy.ndarray): the normalized T2 data array
  """
  exp_T2_data = normalise(expt_T2_data)
  norm_T2_data = np.zeros(expt_T2_data.shape)
  for i in range(expt_T2_data.shape[0]):
    norm_T2_data[i,:] = exp_T2_data[i,:]/exp_T2_data[i,:3].mean()
  return norm_T2_data

# exponential function fit
def simpleExp(T0: float, T1: float) -> float:
    """
    A function that calculates the exponential decay of T0 divided by T1.

    Parameters:
    - T0: The base value used in the exponential calculation.
    - T1: The divisor value used in the exponential calculation.

    Returns:
    - The result of the exponential calculation.
    """
    C = np.exp(-(T0/T1))
    return C
  
# Extract T1 by fitting the data with simple-exponential function
def fit_simpleExp(expt_T1_data: np.ndarray, T0: float, bounds: tuple) -> tuple[float, float]:
    """
    Fit a simple exponential function to the given T1 data using scipy curve fit method.

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
    """
    params = curve_fit(simpleExp, T0, expt_T1_data, bounds = bounds)
    T1 = params[0]
    T1err = np.sqrt(np.diag(params[1]))
    return T1, T1err

# Stretched-exponential function
def stretchExp(T0: float, T2: float, p: float, A: float) -> float:
    """
    A function that calculate Stretched-exponential function, based on the input parameters T0, T2, p, and A.
    
    Parameters:
    - T0 (float): The base value used in the exponential calculation.
    - T2 (float): The divisor value used in the exponential calculation.
    - p (float): The streaching parameter.
    - A (float): The amplitude parameter.
    
    Returns: 
          C (float): The value of Stretched-exponential function.
    """
    C = A*np.exp(-((T0/T2)**p))
    return C

# Extract T2, p, and amplitude by fitting the data with stretched-exponential function
def fit_stretchExp(C: float, T0: float, bounds: tuple) -> tuple[float, float, float, float, float, float]:
    """
    Fit a stretch exponential curve to the given data.

    Parameters:
    - C (float): The data to fit the curve to using scipy curve fit method.
    - T0 (float): The independent variable for the curve fit.
    - bounds (tuple): The bounds for the curve fit. Example: bounds = ([100e-6,1,0.97],[400e-6,2,1.03])

    Returns:
    - Tuple: The fitted parameters T2, p, A, and their respective errors T2err, perr, Aerr.
    """
    params = curve_fit(stretchExp, T0, C, bounds = bounds)
    T2, p, A = params[0]
    T2err, perr, Aerr = np.sqrt(np.diag(params[1]))
    return T2, p, A, T2err, perr, Aerr

# To get final experimental coherence curves/data
def c_expt_data(norm_T2_data: np.ndarray, expt_T1_data: np.ndarray) -> np.ndarray:
  """
  Calculate experimental coherence curves data based on normalized T2 and experimental T1 data.
  
  Parameters:
  - norm_T2_data (np.ndarray): Array of normalized T2 data.
  - expt_T1_data (np.ndarray): Array of experimental T1 data.
  
  Returns:
  - (np.ndarray): Array of experimental coherence curves data.
  """
  expt_c_data = np.zeros((expt_T1_data.shape))
  for i in range(expt_T1_data.shape[0]):
    expt_c_data[i,:] = norm_T2_data[i,:]/np.sqrt(expt_T1_data)[i,:] # norm_T2 over sqrt-T1
  return expt_c_data

# To get fitting of expt_c_data to extract T2 and p parameters values, and fitting curves 
def get_c_expt_fitPar_fitCurves(expt_c_data: np.ndarray, expt_t_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """
  Generate the exponential fit parameters and fit curves for the given experimental data.

  Parameters:
  - expt_c_data (np.ndarray): The experimental concentration data.
  - expt_t_data (np.ndarray): The experimental time data.

  Returns:
  - tuple[np.ndarray, np.ndarray]: A tuple containing the exponential fit parameters and fit curves.
  """
  expt_c_param = np.zeros((expt_c_data.shape[0],3)) 
  expt_c_fit_curves = np.zeros((expt_c_data.shape))
  for i in range(expt_c_data.shape[0]):
    expt_c_param[i,0],expt_c_param[i,1],expt_c_param[i,2],_,_,_ = fit_stretchExp(np.squeeze(expt_c_data[i,:]),expt_t_data, ([100e-6,1,0.97],[400e-6,2,1.03]))
    expt_c_fit_curves[i,:] = stretchExp(expt_t_data,expt_c_param[i,0],expt_c_param[i,1],expt_c_param[i,2])
  return expt_c_param, expt_c_fit_curves

# Fit multiple coherence curves to obtain values of T2 and p (stretching factor)
# Another way to define above function useful for training data generation to check T2 and p distribution
def get_fitPar(c_check: np.ndarray, T_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """
  Generate the T2 values and corresponding parameters for each row in the input array using the fit_stretchExp function.
  
  Parameters:
  - c_check (numpy.ndarray): Input array of shape (n, m) where n is the number of rows and m is the number of columns.
  - T_train (numpy.ndarray): Training data array.
  
  Returns:
  - Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing two numpy arrays. The first array contains the T2 values multiplied by 1e6 and rounded to 1 decimal place. The second array contains the corresponding parameters rounded to 2 decimal places.
  """
  T2_check = []
  p_check = []
  for i in trange(c_check.shape[0]):
    T2_check0, p_check0, _, _, _, _ = fit_stretchExp(c_check[i,:],T_train)
    T2_check.append(T2_check0)
    p_check.append(p_check0)
  return np.round((np.array(T2_check)*1e6),1), np.round((np.array(p_check)),2)

# coherence curve for noisy experiment data along with fitting curves and respective parameters
def get_c_expt_plot(expt_c_data: np.ndarray, expt_t_data: np.ndarray) -> go.Figure:
  """
  Generate a plot of experimental coherence data against evolution time for each qubit. 
  Parameters:
  - expt_c_data (numpy.ndarray): Array of coherence data for each qubit.
  - expt_t_data (numpy.ndarray): Array of evolution time data.
  Returns:
  - go.Figure: A plot displaying the coherence data for each qubit and corresponding fit curves.
  """
  fig = go.Figure()
  for i in range(expt_c_data.shape[0]):
    fig.add_scatter(x=expt_t_data*1e6,y=expt_c_data[i,:],name=f'Qubit {i}',
                  mode="lines+markers",opacity=0.5)
    fig.add_scatter(x=expt_t_data*1e6,y=get_c_expt_fitPar_fitCurves(expt_c_data,expt_t_data)[1][i,:],name=f'{int(get_c_expt_fitPar_fitCurves(expt_c_data,expt_t_data)[0][i,0]*1e6)},'+f'{round(get_c_expt_fitPar_fitCurves(expt_c_data,expt_t_data)[0][i,1],2)}',mode="lines",line=dict(width=3,color='black'),opacity=1)

  fig.update_layout(template=fig_template, width = 700,
                  xaxis = dict(title='Evolution time (\N{greek small letter mu}s)',range=[0,700]),
                  yaxis = dict(title='Coherence'),
                  )
  return fig.show()

# Heatmaps of T1 and T2 experiements
def heatmap(expt_t_data: np.ndarray, expt_c_data: np.ndarray) -> go.Figure:
  """
  Create a heatmap plot based on the experimental transmon data and control data.
  
  Parameters:
  - expt_t_data: The experimental transmon data
  - expt_c_data: The experimental control data
  
  Returns:
  - fig: The generated heatmap plot
  """
  fig = go.Figure()
  fig.add_heatmap(x=expt_t_data*1e6,y=[i for i in range(expt_c_data.shape[0])],z=[expt_c_data[i,:] for i in range(expt_c_data.shape[0])],showscale=True,colorbar=dict(len=0.5,thickness=20,y=0.65))
  fig.update_layout(template=fig_template,
                    width = 700,
                    height = 600,
                    xaxis = dict(title='Evolution time',range=[0,700]),
                    yaxis = dict(title='Qubit no.'))
  return fig.show()

# To save image in pdf format
# pio.write_image(fig,path,format='pdf')

def main():
  # Load experimental data
  expt_data = np.load('/Users/bhavesh/bhavesh/codes/Noise_spectrocopy_ML/data/T1_T2_data.npz',allow_pickle=True)
  print(expt_data.files)
  
  # Extract experimental data
  expt_t_data = expt_data['xval']
  expt_T1_data = expt_data['pop_t1']
  expt_T2_data = expt_data['pop_t2_X32']
  
  # Normalize experimental data and calculate coherence curve
  norm_X32_data = normT2data(expt_T2_data)
  expt_c_data_X32 = c_expt_data(norm_X32_data, expt_T1_data)
  
  # Get fit parameters and fit curves of experimental coherence curve
  expt_c_param_X32, expt_c_fit_curves_X32 = get_c_expt_fitPar_fitCurves(expt_c_data_X32, expt_t_data)
  print(expt_c_param_X32[:,0]*1e6)
  print(expt_c_param_X32[:,1])
  
  # Plot experimental coherence curve
  get_c_expt_plot(expt_c_data_X32, expt_t_data)
  
if __name__ == '__main__':
  main()