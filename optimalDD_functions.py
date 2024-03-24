# Not complete for last function to compare filter function of cpmg and optimal!!

import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import Bounds, minimize, LinearConstraint
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

# %% Generate the Filter function
# function that create CPMG-like pulse timing array
def cpmgFilter(n, Tmax):
    tpi = np.empty([n])
    for i in range(n):
        tpi[i]= Tmax*(((i+1)-0.5)/n)
    return tpi

# Generate filter function for a given evolution time (CPMG)
def getFilter(n,w0,piLength,Tmax):

    tpi = cpmgFilter(n,Tmax)
    f = 0
    for i in range(n):
        f = ((-1)**(i+1))*(np.exp(1j*w0*tpi[i]))*np.cos((w0*piLength)/2) + f

    fFunc = (1/2)*((np.abs(1+((-1)**(n+1))*np.exp(1j*w0*Tmax)+2*f))**2)/(w0**2)
    return fFunc

# Generate filter function for arbitrary pulse sequence
def arbFilter(w0,piLength,tpi,Tmax):
  # print(tpi)
  n = tpi.size
  f = 0
  for i in range(n):
      f = ((-1)**(i+1))*(np.exp(1j*w0*tpi[i]))*np.cos((w0*piLength)/2) + f

  fFunc = (1/2)*((np.abs(1+((-1)**(n+1))*np.exp(1j*w0*Tmax)+2*f))**2)/(w0**2)
  return fFunc

# %%
# Generate decoherence curve corresponding to a noise spectrum under CPMG filter function
def getCoherence(S,w0,T0,n,piLength):
    steps = T0.size
    C_invert = np.empty([S.shape[0],steps,])
    for i in trange(steps):
        integ = getFilter(n,np.squeeze(w0),piLength,T0[i])*S/np.pi
        integ_ans = np.trapz(y=integ,x=np.squeeze(w0))
        C_invert[:,i] = np.exp(integ_ans)
    return C_invert

# Generate decoherence curve corresponding to a noise spectrum under arbitary filter function
def optCoherence(S,w0,piLength,tpi,Tmax):
    integ = arbFilter(w0,piLength,tpi,Tmax)*S/np.pi
    integ_ans = np.trapz(y=integ,x=w0)
    C_invert = np.exp(integ_ans)
    return C_invert

# %%
# Stretched-exponential function
def stretchExp(T0,T2,p,A):
    C = A*np.exp(-((T0/T2)**p))
    return C

# Extract T2, p, and amplitude by fitting the data with stretched-exponential function
def fit_stretchExp(C,T0):
    params = curve_fit(stretchExp, T0, C, bounds=([10e-6,0.9,0.995],[1000e-6,3.2,1.005]))
    T2, p, A = params[0]
    T2err, perr, Aerr = np.sqrt(np.diag(params[1]))
    return T2, p, A, T2err, perr, Aerr

# Fit multiple coherence curves to obtain values of T2 and p (stretching factor)
def get_fitPar(c_check,T_train):
  T2_check = []
  p_check = []
  for i in trange(c_check.shape[0]):
    T2_check0, p_check0, _, _, _, _ = fit_stretchExp(c_check[i,:],T_train)
    T2_check.append(T2_check0)
    p_check.append(p_check0)
  return np.round((np.array(T2_check)*1e6),1),np.round((np.array(p_check)),2)

# %%
# S(w) plot predicted by network
def predicted_s_plot(w_in,s_in):
    fig = go.Figure()
    for i in range(s_in.shape[0]):
        fig.add_scatter(x=1e-6*w_in/(2*np.pi),y=s_in[i,:],line=dict(width=2),opacity=0.8,name=i)

    fig.update_layout(template=fig_template,  title = 'Noise Spectra',width = 1000,xaxis = dict(title='\N{greek small letter omega}/2\N{greek small letter pi} (MHz)',type="log",
                                #  range=[5,50],
                                ),
                    yaxis = dict(title='S(\N{greek small letter omega})',
                                #  range=[-0.1e6,0.1e6],
                                type="log",),)
    return fig

# %% function to evalute w_low_s_low 
'''
s_select = 15
piLength = 48e-9
w_extra = 1001
'''
def w_low_s_low(w_in,w_extra):
    w_low = np.flipud(np.linspace(10,w_in[-1]-10,w_extra))
    s_low = 1e6*np.ones(w_extra)
    return w_low,s_low

# %%
# function to evalute w_extend_s_extend
def w_extend_s_extend(s_in,w_in,s_low,w_low,s_select):
    w_extend = np.concatenate((w_in,w_low))
    s_extend = np.concatenate((s_in[s_select,:],s_low))
    return w_extend,s_extend

# %%
# function to evalute c_extend
def c_extend(s_extend,w_extend,T_in,piLength):
    c_test = getCoherence(np.reshape(s_extend,(1,s_extend.size)),w_extend,T_in,1,piLength)
    return c_test

# %%
# c_val for Custom DD
def customDD(s_extend,w_extend,piLength,x,Tmax):
  c_val = -optCoherence(s_extend,w_extend,piLength,x,Tmax)
  return c_val

# %%
'''
function to evalute C_DD, C_CPMG
data_points = 20
n_plots = 15
n = 8
piLength = 48e-9
'''
def c_dd_c_cpmg(s_in,w_in,s_low,w_low,data_points,n_plots,n,piLength):
    C_DD = np.zeros((n_plots,data_points))
    C_CPMG = np.zeros((n_plots,data_points))
    Tmax = np.linspace(5,300,data_points)*1e-6
    w_extend = w_extend = np.concatenate((w_in,w_low))
    for k in trange (n_plots):
        s_extend = np.concatenate((s_in[k,:],s_low))
        lower_bounds = np.zeros(n)
        for i in range (n):
            lower_bounds[i] = (1/2+i)*piLength

        coeff_matrix = -np.diag(np.ones(n))
        for i in range(n-1):
            coeff_matrix[i, i+1] = 1

        M = coeff_matrix[:-1]
        lower_B = piLength*np.ones(n-1)  

        for j in range(data_points):
            upper_bounds = np.zeros(n)
            for i in range (n):
                upper_bounds[i] = Tmax[j]-(n-(2*i+1)/2)*piLength

            bounds  = Bounds(lower_bounds,upper_bounds)

            upper_B = (Tmax[j]-(n-1)*piLength)*np.ones(n-1)
            linear_constraint = LinearConstraint(M,lower_B,upper_B)

            x0 = cpmgFilter(n,Tmax[j])

            res = minimize(customDD, x0, method='SLSQP',
                    constraints=[linear_constraint], options={'ftol': 1e-9, 'disp': True},
                    bounds=bounds)
            C_DD[k,j] = -customDD(s_extend,w_extend,piLength,res.x,Tmax[j])
            C_CPMG[k,j] = -customDD(s_extend,w_extend,piLength,res.x0,Tmax[j])    
    return C_DD,C_CPMG

# %% 
# c_vals comparison plot btw custom_DD and CPMG_DD 
def c_dd_c_cpmg_plot(C_DD,C_CPMG,Tmax,n_plots):
    fig = go.Figure()
    for i in range(n_plots):
        fig.add_scatter(x=Tmax,y=C_DD[i,:], name = F'C_DD_{i}')
        fig.add_scatter(x=Tmax,y=C_CPMG[i,:], name = F'C_CPMG_{i}')
    fig.update_layout(template=fig_template,  title = 'Coherence Functions',
                    width = 1000,
                    # xaxis = dict(title='\N{greek small letter omega} (MHz)',range=[-6,1],type="log",),
                    xaxis = dict(title = 't'),
                    yaxis = dict(title='C(t)'),
                    )
    return fig.show

# %% To obtain data point C_vals, T_vals and T_vals, T_vals_new function
'''
s_select = 2
n_pi = 8
piLength = 48e-9
data_points = 100
'''
def cpmg_optimal_c_vals(w_low,s_low,w_in,s_in,s_select,n_pi,data_points,piLength):
    n = np.linspace(4,32,n_pi).astype(int)
    Tmax = np.linspace(5,700,data_points)*1e-6
    w_extend = np.concatenate((w_in,w_low))
    s_extend = np.concatenate((s_in[s_select,:],s_low))
    C_vals_new = np.zeros((n_pi,data_points)) # Coherence for optimal DD
    C_vals = np.zeros((n_pi,data_points))     # Coherence for CPMG
    T_vals_new = {}
    T_vals = {}
    for k in trange(n_pi):

        lower_bounds = np.zeros(n[k])
        for i in range (n[k]):
            lower_bounds[i] = (1/2+i)*piLength

        coeff_matrix = -np.diag(np.ones(n[k]))
        for i in range(n[k]-1):
            coeff_matrix[i, i+1] = 1

        M = coeff_matrix[:-1]
        lower_B = piLength*np.ones(n[k]-1)

        for j in range(data_points):
            upper_bounds = np.zeros(n[k])
            for i in range (n[k]):
                upper_bounds[i] = Tmax[j]-(n[k]-(2*i+1)/2)*piLength

            bounds  = Bounds(lower_bounds,upper_bounds)

            upper_B = (Tmax[j]-(n[k]-1)*piLength)*np.ones(n[k]-1)
            linear_constraint = LinearConstraint(M,lower_B,upper_B)

            x0 = cpmgFilter(n[k],Tmax[j])

            res = minimize(customDD, x0, method='SLSQP',
                    constraints=[linear_constraint], options={'ftol': 1e-9, 'disp': True},
                    bounds=bounds)
            # T_vals_new[F'T_vals_new_{n[k]}'] = res.x
            # T_vals[F'T_vals_{n[k]}'] = x0
            T_vals_new.update({F'T_vals_new_{n[k]}_{j}':res.x})
            T_vals.update({F'T_vals_{n[k]}_{j}':x0})
            C_vals_new[k,j] = -customDD(s_extend,w_extend,piLength,res.x,Tmax[j])
            C_vals[k,j] = -customDD(s_extend,w_extend,piLength,res.x0,Tmax[j])
    return C_vals,C_vals_new,T_vals,T_vals_new

# %%
# Comparison plot
def comparison_plot(C_vals,C_vals_new,Tmax,n):
    fig = go.Figure()
    # fig.add_scatter(x=Tmax,y=C_vals_new, name = F'C_DD_{n}')
    # fig.add_scatter(x=Tmax,y=C_vals, name = F'C_CPMG_{n}')
    # fig.add_scatter(x=Tmax,y=C_vals32, name = F'C_CPMG32')

    # fig.add_scatter(x=Tmax,y=100*(C_vals_new/C_vals - 1), name = F'C_CPMG_{n}')
    # fig.add_scatter(x=Tmax,y= 100*(C_vals_new - C_vals), name = F'C_CPMG_{n}')
    for i in range(8):
        fig.add_scatter(x=Tmax,y=100*(C_vals_new[i,:]/C_vals[i,:] - 1), name = F'C_CPMG_{n[i]}')

    fig.update_layout(template=fig_template,  title = 'Comparision_Plots',
                    width = 1000,
                    # xaxis = dict(title='\N{greek small letter omega} (MHz)',range=[-6,1],type="log",),
                    xaxis = dict(title = 't'),
                    # yaxis = dict(title='C(t)'),
                    yaxis = dict(title='Coherence Improvement (%)',
                                #  type="log",
                                #  range=[-6,2],
                                ),
                    )
    return fig.show()

# %%
# CPMG v/s Optimal filter function comparison ******* (This is not complete)
def cpmg_optimal_filter_comp_plot(w_extend,T_vals,T_vals_new): 
    # f_test = arbFilter(w_extend,piLength,cpmgFilter(n,Tmax),Tmax)
    # f_optimal = arbFilter(w_extend,piLength,result.x,Tmax)
    fig = go.Figure()
    fig.add_scatter(x=1e-6*w_extend/(2*np.pi),y=T_vals_new['T_vals_new_8_33'])
    fig.add_scatter(x=1e-6*w_extend/(2*np.pi),y=T_vals['T_vals_new_8_33'])
    fig.update_layout(template=fig_template,  title = 'Filter Functions',
                    width = 1000,
                    xaxis = dict(title='\N{greek small letter omega} (MHz)',range=[-6,1],type="log",),
                    yaxis = dict(title='F(\N{greek small letter omega}t)/\N{greek small letter omega}<sup>2</sup>'),
                    )
    return fig