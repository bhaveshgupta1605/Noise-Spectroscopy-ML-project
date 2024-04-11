# Maybe Incomplete!!
'''
# To evaluate a decoherence curve from an underlying noise spectrum
Siddharth Dhomkar, Department of Physics, IIT Madras

**Based on**:

D.F. Wise, J.J.L. Morton, and S. Dhomkar, “Using deep learning to understand and mitigate the qubit noise environment”, PRX Quantum 2, 010316 (2021)

*   https://doi.org/10.1103/PRXQuantum.2.010316
*   http://dx.doi.org/10.1103/PhysRevApplied.18.024004
'''
# %% Import library files, modules and packages
import numpy as np # Module for scientific computing
from scipy.interpolate import interp1d # Module for interpolation
from scipy.optimize import curve_fit # Module for fitting a function to data
from tqdm.notebook import trange # Provides a progress bar for loops in Jupyter notebooks
from plotly import graph_objs as go # Data visualization from Plotly
import random # For generating random numbers

from expt_data_analysis import get_fitPar # Importing the expt_data_analysis.py file

# %% Figure Template
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

'''
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
'''

# %% Generate the Filter function
# function that create CPMG-like pulse timing array
def cpmgFilter(n: int, Tmax: float) -> np.ndarray:
  """
  Generates an array of time points for a CPMG filter.
  Parameters:
  - n (int): the number of time points
  - Tmax (float): the maximum time
  Returns:
  - np.ndarray: an array of time points
  """
  tpi = np.empty([n])
  for i in range(n):
    tpi[i]= Tmax*(((i+1)-0.5)/n)
  return tpi

# Generate filter function for a given pulse sequence
def getFilter(n: int, w0: float, piLength: float, Tmax: float) -> np.ndarray:
  """
  Returns the filter function based on the given parameters.
  Parameters:
  - n (int): The number of iterations.
  - w0 (float): The angular frequency.
  - piLength (float): The length of pi.
  - Tmax (float): The maximum time.
  Returns:
  - np.ndarray: The filter function.
  """
  tpi = cpmgFilter(n,Tmax)
  f = 0
  for i in range(n):
    f = ((-1)**(i+1))*(np.exp(1j*w0*tpi[i]))*np.cos((w0*piLength)/2) + f
  fFunc = (1/2)*((np.abs(1+((-1)**(n+1))*np.exp(1j*w0*Tmax)+2*f))**2)/(w0**2)
  return fFunc

# %% Generate the noise spectra
# Function to generate the noise spectrum
def transmon_noise(T2: np.ndarray, w: np.ndarray) -> np.ndarray:
  """
  Calculates the transmon noise for the given parameters.

  :Param
  - T2: numpy array containing T2 values
  - w: numpy array containing w values

  Return:
  - numpy array of smoothed transmon noise, numpy array of w values
  """
  alpha0 = 0
  alpha1 = 1
  alpha2 = 2
  bound_cutoff = 5500
  bound_threshold = 10000

  A0 = np.reshape(((2**alpha0)*((np.pi/T2)**(alpha0+1))),[T2.size,1])
  A1 = np.reshape(((2**alpha1)*((np.pi/T2)**(alpha1+1))),[T2.size,1])
  A2 = np.reshape(((2**alpha2)*((np.pi/T2)**(alpha2+1))),[T2.size,1])
  w = np.reshape(w,[1,w.size])

  width = 2000
  # width = 500
  s_smooth = np.zeros((T2.size,w.size-width+1))
  for i in trange(T2.size):
    wc_check = np.random.random_sample()
    if wc_check<0.25:
      # cut = -np.sort(np.array(random.sample(range(bound_cutoff,w.size),2)))
      cut1 = -np.array(random.sample(range(bound_cutoff,bound_threshold),1))
      cut2 = -np.array(random.sample(range(-cut1[0]+1,w.size),1))
      cut = np.squeeze(np.array([cut1,cut2]))
      s0 = np.squeeze(A0[i,:]/w[:,cut[0]:]**alpha0)
      s1 = np.squeeze(A1[i,:]/w[:,cut[1]:cut[0]]**alpha1)
      s1 = np.reshape(s1,(s1.size,))
      s1 = (s0[0]/s1[-1])*s1
      s2 = np.squeeze(A2[i,:]/w[:,:cut[1]]**alpha2)
      s2 = np.reshape(s2,(s2.size,))
      s2 = (s1[0]/s2[-1])*s2
      s =  0.65*np.concatenate((s2,s1,s0),axis=0)
    else:
      # cut = -np.array(random.sample(range(bound_cutoff,w.size),1))
      cut = -np.array(random.sample(range(bound_cutoff,bound_threshold),1))
      s0 = np.squeeze(A0[i,:]/w[:,cut[0]:]**alpha0)
      s1 = np.squeeze(A1[i,:]/w[:,:cut[0]]**alpha1)
      s1 = np.reshape(s1,(s1.size,))
      s1 = (s0[0]/s1[-1])*s1
      s =  0.65*np.concatenate((s1,s0),axis=0)

    s_smooth[i,:] = moving_average(s,width)
  # np.squeeze(w)[w.size-width-1:]
  return s_smooth, np.squeeze(w)[:w.size-width+1]

# function to obtain moving average
def moving_average(x: np.ndarray, width: int) -> np.ndarray:
  """
  Calculate the moving average of a 1D array.
  Parameters:
    x (array): The input 1D array.
    width (int): The width of the moving window.

  Returns:
    array: The moving average of the input array.
  """
  return np.convolve(x, np.ones(width),'valid')/width

# %%
# function which generate coherence curve corresponding to a noise spectrum
def getCoherence(S: np.ndarray, w0: np.ndarray, T0: np.ndarray, n: int, piLength: int) -> np.ndarray:
  """
  Get the coherence values for a given input signal S.
  Parameters:
  - S (np.ndarray): The input signal.
  - w0 (np.ndarray): The frequency vector.
  - T0 (np.ndarray): Time vector.
  - n (int): Number of pulse.
  - piLength (int): Length of X-pi pulse.
  Returns:
  - np.ndarray: Coherence values for the input Spectrum signal.
  """
  steps = T0.size
  C_invert = np.empty([S.shape[0],steps,])
  for i in trange(steps):
    integ = getFilter(n,np.squeeze(w0),piLength,T0[i])*S/np.pi
    integ_ans = np.trapz(y=integ,x=np.squeeze(w0))
    C_invert[:,i] = np.exp(integ_ans)
  return C_invert


# %% Now training data generation without addition of random gussian noise
# function which evalute noise spectrum, coherence and fitting parameters as smooth training data
def SmoothTrainData(T_in: np.ndarray, T2: np.ndarray,w: np.ndarray, n_pulse: int, piLength: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
  """
  A function that takes in training data and parameters, and returns a tuple of arrays representing the processed data and fit parameters.
  
  Parameters:
  - T_in (numpy.ndarray): Time vector
  - T2 (numpy.ndarray): possible T2 values as parameters
  - w (numpy.ndarray): omega vector
  - n_pulse (int): Number of pulses
  - piLength (int): Duration of the pi-pulse
  
  Example:
  - T_in = t = np.geomspace(1.8e-6,750.05e-6,451) 
  - T2 = np.linspace(100e-6,370e-6,15001)
  - w = np.flipud(np.logspace(2.0,10.0,16001)) 
  - n_pulse = 32 
  - piLength = 48e-9  
  
  Returns:
  - c_in (numpy.ndarray): Coherence values
  - s_in (numpy.ndarray): Noise spectrum
  - w_in (numpy.ndarray): Frequency vector
  - fit_par (tuple[numpy.ndarray, numpy.ndarray]): Fit parameters
  
  Check out the shape of the returned arrays for more details:
  - c_in, s_in, w_in, fit_par = SmoothTrainData(T_in, T2, w, n_pulse, piLength)
  - c_in.shape, s_in.shape, w_in.shape, fit_par[0].shape, fit_par[1].shape
  """
  s_in, w_in = transmon_noise(T2,w) # input spectrum and omega vector
  c_in = getCoherence(s_in,w_in,T_in,n_pulse,piLength) # Evaluate decoherence curves
  fit_par = get_fitPar(c_in,T_in)
  return c_in, s_in, w_in, fit_par

# function plots histogram of T2 and p parameter distribution in our smooth training data generated
def T2_p_distribution(fit_par: tuple[np.ndarray, np.ndarray]) -> tuple[go.Figure, go.Figure]:
  """
  Generate two bar plots showing the distribution of T2 values and stretch factors based on the input fit parameters.
  
  Parameters:
  fit_par (tuple[np.ndarray, np.ndarray]): A tuple containing two numpy arrays representing the fitted parameters.
  
  Returns:
  tuple[go.Figure, go.Figure]: A tuple containing two plotly figures representing the T2 distribution and p distribution.
  """
  hist0,x0 =np.histogram(fit_par[0],bins=51,range=[50,700],density=False)
  hist1,x1 =np.histogram(fit_par[1],bins=51,range=[0.97,3],density=False)
  fig1 = go.Figure()
  fig1.add_trace(go.Bar(x=x0,y=hist0,opacity=0.75,marker_color='royalblue'))
  fig1.update_layout(bargap=0)
  fig1.update_layout(template=fig_template)
  fig1.layout.xaxis.title = 'Fitted T<sub>2</sub> (\N{greek small letter mu}s)'
  fig1.layout.yaxis.title = 'Number of datasets'
  fig1.layout.title = 'T<sub>2</sub> Distribution'

  fig2 = go.Figure()
  fig2.add_trace(go.Bar(x=x1,y=hist1,opacity=0.75,marker_color='royalblue'))
  fig2.update_layout(bargap=0)
  fig2.update_layout(template=fig_template)
  fig2.layout.xaxis.title = 'Stretch factor'
  fig2.layout.yaxis.title = 'Number of datasets'
  fig2.layout.title = 'p Distribution'
  return fig1, fig2

# %% Now addition of random noise to generate our final training data
# For data interpolation
def interpData(x: np.ndarray, y: np.ndarray, xNew: np.ndarray) -> np.ndarray:
    f_interp = interp1d(x,y)
    yNew = f_interp(xNew)
    return yNew
'''
For preparing training data: Add random noise
Run this cell multiple times to generate sets with different random noise but same underlying curves
Actual time vector used in the experiment (Should be smaller than t)
Actual frequency vector used for training (Should be smaller than w)
w_train = np.flipud(np.geomspace(140e3,55e6,501)) # for example
'''
#%%
# Interpolate experimental coherence data if the experimental and the simulated time vectors are different
def interpolte_c_expt_data(expt_T2_data: np.ndarray, expt_t_data: np.ndarray, T_train: np.ndarray) -> np.ndarray:
  """
  Interpolates experimental data to generate c_exp values for given T_train values.
  
  Args:
  - expt_T2_data (np.ndarray): Array of experimental T2 data.
  - expt_t_data (np.ndarray): Array of experimental t data.
  - T_train (np.ndarray): Array of T values for training.

  Returns:
  - np.ndarray: Array of Interpolated c_exp values.
  """
  c_exp = np.zeros((expt_T2_data.shape[0],T_train.size))
  for i in range(expt_T2_data.shape[0]):
    c_exp[i,:] = interpData(np.round(expt_t_data['xval'],10),expt_T2_data[0,:],np.round(T_train,10))
  return c_exp
# T_train = expt_data['xval'] # To avoide above issue, keep T_train and T_expt same

#%%
def prepare_trainData(c_in: np.ndarray, T_in: np.ndarray, T_train: np.ndarray, noiseMax: float = 0.03) -> np.ndarray:
  """
  Prepare the training data by interpolating the input data to the training time points.
  
  Parameters:
  - c_in: Input data for interpolation
  - T_in: Time points corresponding to the input data
  - T_train: Time points for training data
  - noiseMax: Maximum noise level to add (default is 0.03)
  
  Returns:
  - c_train: Interpolated training data with added noise
  """
  c_train = interpData(T_in,c_in,T_train)
  for i in trange(c_in.shape[0]):
    c_train[i,:] = c_train[i,:] + np.random.normal(0,noiseMax,size=c_train.shape[1])
  return c_train

# For processing the experimental data as training data: interpolation, and random cut-off
def prepare_expData(c_in: np.ndarray, T_in: np.ndarray, T_train: np.ndarray, cutOff: float = 0.03) -> np.ndarray:
  """
  Generate a prediction for experimental data based on input data and training data.

  Parameters:
    c_in (numpy array): Array of input data.
    T_in (numpy array): Array of input temperatures.
    T_train (numpy array): Array of training temperatures.
    cutOff (float, optional): Cutoff value for predictions. Default is 0.03.

  Returns:
    c_predict (numpy array): Predicted values for the experimental data.
  """
  if np.max(T_train) < np.max(T_in):
    c_predict = interpData(T_in,c_in,T_train)
    for i in trange(c_in.shape[0]):
      cut = np.squeeze(np.argwhere(c_predict[i,:]<=cutOff))
      if cut.size > 1:
        c_predict[i,cut[0]-1:] = 0
      elif cut.size == 1:
        c_predict[i,cut-1:] = 0
  else:
    T_train1 = np.squeeze(T_train[np.argwhere(T_train<=np.max(T_in))])
    zero_size = T_train.size - T_train1.size
    c_predict = interpData(T_in,c_in,T_train1)
    for i in trange(c_in.shape[0]):
      cut = np.squeeze(np.argwhere(c_predict[i,:]<=cutOff))
      if cut.size > 1:
        c_predict[i,cut[0]-1:] = 0
      elif cut.size == 1:
        c_predict[i,cut-1:] = 0
    c_predict = np.concatenate((c_predict,np.zeros((c_in.shape[0],zero_size))),axis=1)
  return c_predict

# To get our final noisy training data (as, c_train and S_train) to feed into our netowrk to train
def NoisyTrainData(c_in: np.ndarray, s_in: np.ndarray, w_in: np.ndarray, T_in: np.ndarray, T_train: np.ndarray, w_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """
  Generates noisy training data based on the input parameters.

  Parameters:
    c_in (numpy array): Input c values.
    s_in (numpy array): Input s values.
    w_in (numpy array): Input w value.
    T_in (numpy array): Input T values.
    T_train (numpy array): Training T values.
    w_train (numpy array): Training w values.

  Returns:
    c_train (numpy array): Noisy training c values.
    s_train (numpy array): Noisy training s values.
  """
  c_train = np.zeros((5*c_in.shape[0], T_train.size))
  s_train = np.tile(interpData(w_in, s_in, w_train),(5, 1))

  for i in range(5):
    print(i*c_in.shape[0])
    c_train[i*c_in.shape[0]:(i+1)*c_in.shape[0],:] = prepare_trainData(c_in, T_in, T_train)
  return c_train, s_train

# %% Alternate functions to generate final training data
def generate_final_data(c_data,T_in,s_data,w0,T_train,w_train):
  nnps = 6 #-- noise number per sample
  c_train_1set = prepare_trainData(c_data, T_in, T_train)
  s_train_1set = interpData( w0, s_data, w_train )
  d1 = np.shape(c_train_1set)[0]
  d2 = np.shape(c_train_1set)[1]
  d3 = np.shape(s_train_1set)[1]
  c_train_final = np.zeros((d1*nnps,d2))
  s_train_final = np.zeros((d1*nnps,d3))
  for i in range(nnps):
    c_train_1set = prepare_trainData( c_data, T_in, T_train, noiseMax = 0.03)
    c_train_final[i*d1:(i+1)*d1,:] = c_train_1set
    s_train_final[i*d1:(i+1)*d1,:] = s_train_1set

  return c_train_final, s_train_final

def spectrum_extend(exp_predict,w_train,w_new):
  w_lowSize = np.argwhere(w_new<w_train.min()).size
  w_lowArg = np.argwhere(w_new<w_train.min())[0]

  s_extend = np.zeros((exp_predict.shape[0],w_new.size))
  s_extend[:,int(w_lowArg):] = np.repeat(np.mean(exp_predict[:,-4:],axis=1),
                                        int(w_lowSize)).reshape(exp_predict.shape[0],int(w_lowSize))
  for i in range(exp_predict.shape[0]):
    s_extend[i,:int(w_lowArg)] = interpData(w_train,exp_predict[i,:],w_new[:int(w_lowArg)])
  return s_extend

def main():
  pass

if __name__ == '__main__':
  main()
