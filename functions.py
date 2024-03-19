import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline
from tqdm.notebook import trange
import random

# Normalise experimental decoherence data from 1 to 0
def normalise(data):
  return (data-0.5)/0.5
   
# exponential function fit
def simpleExp(T0,T1):
    C = np.exp(-(T0/T1))
    return C
  
# Extract T1 by fitting the data with simple-exponential function
def fit_simpleExp(signal,T0):
    params = curve_fit(simpleExp, T0, signal, bounds=([10e-6],[1000e-6]))
    T1 = params[0]
    T1err = np.sqrt(np.diag(params[1]))
    return T1, T1err

# Stretched-exponential function fit
def stretchExp(T0,T2,p,A):
    C = A*np.exp(-((T0/T2)**p))
    return C

# Extract T2, p, and amplitude by fitting the data with stretched-exponential function
def fit_stretchExp(C,T0):
    params = curve_fit(stretchExp, T0, C, bounds=([100e-6,1,0.97],[400e-6,2,1.03]))
    T2, p, A = params[0]
    T2err, perr, Aerr = np.sqrt(np.diag(params[1]))
    return T2, p, A, T2err, perr, Aerr

# Create CPMG-like pulse timing array
def cpmgFilter(n, Tmax):
    tpi = np.empty([n])
    for i in range(n):
        tpi[i]= Tmax*(((i+1)-0.5)/n)
    return tpi

# Generate filter function for a given pulse sequence
def getFilter(n,w0,piLength,Tmax):
    tpi = cpmgFilter(n,Tmax)
    f = 0
    for i in range(n):
        f = ((-1)**(i+1))*(np.exp(1j*w0*tpi[i]))*np.cos((w0*piLength)/2) + f

    fFunc = (1/2)*((np.abs(1+((-1)**(n+1))*np.exp(1j*w0*Tmax)+2*f))**2)/(w0**2)
    return fFunc

# Function to generate the noise spectrum
def transmon_noise(T2,w):
  alpha0 = 0
  alpha1 = 1
  alpha2 = 2
  # bound_cutoff = 1800
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

# Moving average
def moving_average(x, width):
    return np.convolve(x, np.ones(width), 'valid') / width

# Generate coherence curve corresponding to a noise spectrum
def getCoherence(S,w0,T0,n,piLength):
    steps = T0.size
    C_invert = np.empty([S.shape[0],steps,])
    for i in trange(steps):
        integ = getFilter(n,np.squeeze(w0),piLength,T0[i])*S/np.pi
        integ_ans = np.trapz(y=integ,x=np.squeeze(w0))
        C_invert[:,i] = np.exp(integ_ans)
    return C_invert

# Fit multiple coherence curves to obtain values of T2 and p (stretching factor)
def get_fitPar(c_check,T_train):
  T2_check = []
  p_check = []
  for i in trange(c_check.shape[0]):
    T2_check0, p_check0, _, _, _, _ = fit_stretchExp(c_check[i,:],T_train)
    T2_check.append(T2_check0)
    p_check.append(p_check0)
  return np.round((np.array(T2_check)*1e6),1),np.round((np.array(p_check)),2)

# For data interpolation
def interpData(x,y,xNew):
    f_interp = interp1d(x,y)
    yNew = f_interp(xNew)
    return yNew

# For preparing training data: Add random noise
# Run this cell multiple times to generate sets with different random noise but same underlying curves
def prepare_trainData(c_in,T_in,T_train,noiseMax=0.03):
  c_train = interpData(T_in,c_in,T_train)
  for i in trange(c_in.shape[0]):
    c_train[i,:] = c_train[i,:] + np.random.normal(0,noiseMax,size=c_train.shape[1])
  return c_train

# For processing the experimental data as training data: interpolation, and random cut-off
def prepare_expData(c_in,T_in,T_train,cutOff=0.03):
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
		c_train_1set = prepare_trainData( c_data, T_in, T_train, noiseMax=0.03)
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