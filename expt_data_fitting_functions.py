import numpy as np
from scipy.optimize import curve_fit
from tqdm.notebook import trange
from plotly import graph_objs as go

# Figure Template
fig_template = go.layout.Template()
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
Import experiment data as npz file
expt_data = np.load('path',allow_pickle=True)
expt_t_data = expt_data['xval'] ---->shape = (time array,) = (*,)
expt_T1_data=expt_data['pop_t1']-->shape =(qubit number, population of |1> at corresponding time array)=($,*)
expt_T2_data=expt_data['pop_t2']-->shape =(qubit number, population of |0> at corresponding time array)=($,*)
expt_data.files ---> should contain 'pop_t1','pop_t2' and 'xval' data files
'''

# Normalise experimental decoherence data from 1 to 0
def normalise(data):
  return (data-0.5)/0.5

# Normalisation of T2_X32 signals
def normT2data(expt_T2_data):
  exp_T2_data = normalise(expt_T2_data)
  norm_T2_data = np.zeros(expt_T2_data.shape)
  for i in range(expt_T2_data.shape[0]):
    norm_T2_data[i,:] = exp_T2_data[i,:]/exp_T2_data[i,:3].mean()
  return norm_T2_data

# exponential function fit
def simpleExp(T0,T1):
    C = np.exp(-(T0/T1))
    return C
  
# Extract T1 by fitting the data with simple-exponential function
def fit_simpleExp(expt_T1_data,T0,bounds):
    params = curve_fit(simpleExp, T0, expt_T1_data,bounds) # for example, bounds=([10e-6],[1000e-6]) in sec
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

# To get final experimental coherence curves/data
def c_expt_data(norm_T2_data,expt_T1_data):
  expt_c_data = np.zeros((expt_T1_data.shape))
  for i in range(expt_T1_data.shape[0]):
    expt_c_data[i,:] = norm_T2_data[i,:]/np.sqrt(expt_T1_data)[i,:] # norm_T2 over sqrt-T1
  return expt_c_data

# To get fitting of expt_c_data to extract T2 and p values, and fitting curves 
def get_c_expt_fitPar_fitCurves(expt_c_data,expt_t_data):
  expt_c_param = np.zeros((expt_c_data.shape[0],3)) 
  expt_c_fit_curves = np.zeros((expt_c_data.shape))
  for i in range(expt_c_data.shape[0]):
    expt_c_param[i,0],expt_c_param[i,1],expt_c_param[i,2],_,_,_ = fit_stretchExp(np.squeeze(expt_c_data[i,:]),expt_t_data)
    expt_c_fit_curves[i,:] = stretchExp(expt_t_data,expt_c_param[i,0],expt_c_param[i,1],expt_c_param[i,2])
  return expt_c_param,expt_c_fit_curves

# Fit multiple coherence curves to obtain values of T2 and p (stretching factor)
# Another way to define above function useful for training data generation to check T2 and p distribution
def get_fitPar(c_check,T_train):
  T2_check = []
  p_check = []
  for i in trange(c_check.shape[0]):
    T2_check0, p_check0, _, _, _, _ = fit_stretchExp(c_check[i,:],T_train)
    T2_check.append(T2_check0)
    p_check.append(p_check0)
  return np.round((np.array(T2_check)*1e6),1),np.round((np.array(p_check)),2)

# coherence curve for noisy experiment data along with fitting curves and respective parameters
def get_c_expt_plot(expt_c_data,expt_t_data):
  fig = go.Figure()
  for i in range(expt_c_data.shape[0]):
    fig.add_scatter(x=expt_t_data*1e6,y=expt_c_data[i,:],name=f'Qubit {i}',
                  mode="lines+markers",opacity=0.5)
    fig.add_scatter(x=expt_t_data*1e6,y=get_c_expt_fitPar_fitCurves(expt_c_data,expt_t_data)[1][i,:],name=f'{int(get_c_expt_fitPar_fitCurves(expt_c_data,expt_t_data)[0][i,0]*1e6)},'+f'{round(get_c_expt_fitPar_fitCurves(expt_c_data,expt_t_data)[0][i,1],2)}',mode="lines",line=dict(width=3,color='black'),opacity=1)

  fig.update_layout(template=fig_template, width = 700,
                  xaxis = dict(title='Evolution time (\N{greek small letter mu}s)',range=[0,700]),
                  yaxis = dict(title='Coherence'),
                 )
  return fig

# Heatmaps of T1 and T2 experiements
def heatmap(expt_t_data,expt_c_data):
  fig = go.Figure()
  fig.add_heatmap(x=expt_t_data*1e6,y=[i for i in range(expt_c_data.shape[0])],z=[expt_c_data[i,:] for i in range(expt_c_data.shape[0])],showscale=True,colorbar=dict(len=0.5,thickness=20,y=0.65))
  fig.update_layout(template=fig_template,
                    width = 700,
                    height = 600,
                    xaxis = dict(title='Evolution time',range=[0,700]),
                    yaxis = dict(title='Qubit no.'))
  return fig

# To save image in pdf format
# pio.write_image(fig,path,format='pdf')