import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from tqdm import trange

# For data interpolation
def interpData(x,y,xNew):
	f_interp = interp1d(x,y)
	yNew = f_interp(xNew)
	return yNew

# For preparing training data: Add random noise, then replace low values with zeros
# Run this cell multiple times to generate sets with different random noise but same underlying curves

def prepare_trainData(c_in,T_in,T_train,noiseMax=0.02):
  c_train = interpData(T_in,c_in,T_train)
  for i in range(c_in.shape[0]):
    c_train[i,:] = c_train[i,:] + np.random.normal(0,noiseMax,size=c_train.shape[1])
  return c_train

def generate_final_data(c_data,T_in,s_data,w0,T_train,w_train):
	nnps = 6 #-- noise number per sample
	c_train_1set = prepare_trainData( c_data, T_in, T_train )
	s_train_1set = interpData( w0, s_data, w_train )
	d1 = np.shape( c_train_1set )[0]
	d2 = np.shape( c_train_1set )[1]
	d3 = np.shape( s_train_1set )[1]
	c_train_final = np.zeros( ( d1*nnps, d2 ) )
	s_train_final = np.zeros( ( d1*nnps, d3 ) )
	for i in range(nnps):
		c_train_1set = prepare_trainData( c_data, T_in, T_train, noiseMax=0.02 )
		c_train_final[i*d1:(i+1)*d1,:] = c_train_1set
		s_train_final[i*d1:(i+1)*d1,:] = s_train_1set

	return c_train_final, s_train_final

# %%
# Create CPMG-like pulse timing array

def cpmgFilter(n, Tmax):
    tpi = np.empty([n])
    for i in range(n):
        tpi[i]= Tmax*(((i+1)-0.5)/n)
    return tpi


# %%
# Generate filter function for a given pulse sequence

def getFilter(n,w0,piLength,Tmax):
    tpi = cpmgFilter(n,Tmax)
    f = 0    
    for i in range(n):
        f = ((-1)**(i+1))*(np.exp(1j*w0*tpi[i]))*np.cos((w0*piLength)/2) + f

    fFunc = (1/2)*(( np.abs(1+((-1)**(n+1))*np.exp(1j*w0*Tmax)+2*f) )**2)/(w0**2)
    return fFunc


# %%
# Generate decoherence curve corresponding to a noise spectrum (input shape = variable1.size x w.size)

def getCoherence(S,w0,T0,n,piLength):
    steps = T0.size
    C_invert = np.empty([S.shape[0],steps,])
    for i in range(steps):
        integ = getFilter(n,np.squeeze(w0),piLength,T0[i])*S/np.pi
        integ_ans = np.trapz(y=integ,x=np.squeeze(w0))
        C_invert[:,i] = np.exp(integ_ans)
    return C_invert    

def spectrum_extend(exp_predict,w_train,w_new):
  w_lowSize = np.argwhere(w_new<w_train.min()).size
  w_lowArg = np.argwhere(w_new<w_train.min())[0]

  s_extend = np.zeros((exp_predict.shape[0],w_new.size))
  s_extend[:,int(w_lowArg):] = np.repeat(np.mean(exp_predict[:,-4:],axis=1),
                                        int(w_lowSize)).reshape(exp_predict.shape[0],int(w_lowSize))
  for i in range(exp_predict.shape[0]):
    s_extend[i,:int(w_lowArg)] = interpData(w_train,exp_predict[i,:],w_new[:int(w_lowArg)])
  return s_extend