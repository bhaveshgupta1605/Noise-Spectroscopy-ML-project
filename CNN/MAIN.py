from tensorflow.keras import models, optimizers, callbacks, Sequential
#import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
import numpy as np
import time
from network_functions import get_model
from data_generation_functions import generate_final_data
import sys 
import os

import matplotlib.pyplot as plt 
import matplotlib
matplotlib.rcParams['figure.dpi']=300


#=============================================
#== COMMAND LINE PARAMETERS
#=============================================

#-- network structure
FILTERS=40
KERNEL_SIZE=5
DROPOUT_RATE=0.05
POOL_SIZE=2

#-- training hyperparamters
BATCH_SIZE=64
EPOCHS=20
INITIAL_LR=1e-3
MIN_LR=1e-6
RED_FACTOR=0.5
DROPOUT_RATE=0.05
ATIENCE=6
MIN_DELTA=0.5
NB=0
VERBOSE=True
NET_TYPE=1

n = len(sys.argv)

for i in range(1, n):
	if isinstance( sys.argv[i], str ):
		if (sys.argv[i]=='--batch_size'):
			BATCH_SIZE=int( sys.argv[i+1] )
		elif (sys.argv[i]=='--epochs'):
			EPOCHS=int(sys.argv[i+1])
		elif (sys.argv[i]=='--filters'):
			FILTERS=int(sys.argv[i+1])
		elif (sys.argv[i]=='--kernel_size'):
			KERNEL_SIZE=int(sys.argv[i+1])
		elif (sys.argv[i]=='--initial_lr'):
			INITIAL_LR=float(sys.argv[i+1])
		elif (sys.argv[i]=='--min_lr'):
			MIN_LR=float(sys.argv[i+1])
		elif (sys.argv[i]=='--min_delta'):
			MIN_DELTA=float(sys.argv[i+1])
		elif (sys.argv[i]=='--patience'):
			PATIENCE=float(sys.argv[i+1])
		elif (sys.argv[i]=='--nb'):
			NB=int(sys.argv[i+1])
		elif (sys.argv[i]=='--verbose'):
			if int(sys.argv[i+1])==1:
				VERBOSE=True
			else:
				VERBOSE=False
		elif (sys.argv[i]=='--net_type'):
			NET_TYPE=int(sys.argv[i+1])

print('=============================')
print('-- BATCH_SIZE  = ', BATCH_SIZE)
print('-- EPOCHS      = ', EPOCHS)
print('-- FILTERS     = ', FILTERS)
print('-- KERNEL_SIZE = ', KERNEL_SIZE)
print('-- NET_TYPE    = ', NET_TYPE)
print('-- INITIAL_LR  = ', INITIAL_LR)
print('-- MIN_LR      = ', MIN_LR)
print('-- MIN_DELTA   = ', MIN_DELTA)
print('-- PATIENCE    = ', PATIENCE)
print('=============================')

#=============================================
#== IMPORTING THE DATA
#=============================================

#== import the data
data = np.load("data/Mar6_x32_noisy.npz")

c_data = data['c_in'] 
T_in = data['T_in']    # Time vector for data generation 
s_data = data['s_in'] 
w0 = data['w_in']           # Omega vector for data generation
T_train = data['T_train']      # Time vector for training data (based on the experimental data)
w_train = data['w_train']      # Omega vector for training data
print('-- data loaded')


#== format the data for the training stage
c_train, s_train = \
generate_final_data( c_data, T_in, s_data, w0, T_train, w_train )

x_train, x_test, y_train, y_test = train_test_split( c_train, s_train, test_size=0.15)


#print('-- data split for training:')
#print("  x_train = ",np.shape(x_train))
#print("  y_train = ",np.shape(y_train))
#print("  x_test = ",np.shape(x_test))
#print("  y_test = ",np.shape(y_test))
#print("  w0 = ",np.shape(w0))
#print("  w_train = ",np.shape(w_train))


#=============================================
#== create the neural, and compile the network
#=============================================

X_TRAIN_SIZE = np.shape(x_train)[-1]

model = get_model( filter_nb=FILTERS, kernel_size=KERNEL_SIZE, pool_size=POOL_SIZE,\
                  dropout_rate=DROPOUT_RATE, xtrain_size=X_TRAIN_SIZE, net_type=NET_TYPE )  #-- create model

#-- prepare the model and the callback function
reduce_lr = callbacks.ReduceLROnPlateau( monitor="loss", \
										factor=RED_FACTOR,\
										patience=PATIENCE,\
										verbose=VERBOSE,\
										mode="auto",\
										min_delta=MIN_DELTA,\
										cooldown=0,\
										min_lr=MIN_LR)  #-- define LR reduction strategy

#-- define optimizer
opt = optimizers.legacy.Adam(learning_rate=INITIAL_LR)  #-- define optimizer

#-- compile
model.compile(loss='MAPE', optimizer=opt)  #-- compilation

#=============================================
#== training
#=============================================

print('-- beginning fit')
t1 = time.time()
history_ = model.fit( x_train, y_train, BATCH_SIZE, epochs=EPOCHS,\
                        validation_data=(x_test, y_test), verbose=VERBOSE, callbacks=[reduce_lr])

final_accuracy = np.round(history_.history['val_loss'][-1],2)
print('-- fit complete')
print('-- time taken=', time.time() - t1)
print('-- final accuracy=', final_accuracy)

#-- define useful paramchar
paramchar_no_fil_no_ker="dr"+str(DROPOUT_RATE)\
            +"_ps="+str(POOL_SIZE)+'_LRini='+str(INITIAL_LR)+'_LRmin='+str(MIN_LR)\
            +'_bs='+str(BATCH_SIZE)+"_ep="+str(EPOCHS)+"_nt="+str(NET_TYPE)
paramchar=str( np.round(history_.history['val_loss'][-1],2) )\
			+"_fil="+str(FILTERS)+"_ker="+str(KERNEL_SIZE)+"_dr"+str(DROPOUT_RATE)\
            +"_ps="+str(POOL_SIZE)+'_LRini='+str(INITIAL_LR)+'_LRmin='+str(MIN_LR)\
            +'_bs='+str(BATCH_SIZE)+"_ep="+str(EPOCHS)+"_nt="+str(NET_TYPE)
print("-- paramchar = "+paramchar)

#-- save the model
model.save( 'TRAINED_NETWORKS/MODEL_'+paramchar, overwrite=True)

#filename = 'TRAINED_NETWORKS/accuracy_'+paramchar_no_fil_no_ker+'.txt'
#if os.path.exists(filename):
#    append_write = 'a' # append if already exists
#else:
#    append_write = 'w' # make a new file if not
#
#f = open( filename, append_write )
#f.write( str(FILTERS) +'    '+ str(KERNEL_SIZE) + '    ' + str( np.round(history_.history['val_loss'][-1],2) ) + '\n' )
#f.close()

#-- plot the history

plt.subplot(1, 1, 1)
plt.axhline(y = final_accuracy, color = 'black', dashes=[2,2,2,2])
plt.title('Final accuracy = '+str(final_accuracy))
plt.plot( np.arange( 0, EPOCHS ) , history_.history['val_loss'], label='validation' )
plt.plot( np.arange( 0, EPOCHS ) , history_.history['loss'], label='training' )
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim(-1,100)
plt.legend()
plt.savefig( 'TRAINED_NETWORKS/VAL_ACC_HISTORY_'+paramchar+'.pdf', format='pdf' )
plt.close()


#=============================================
#== testting the model
#=============================================

#-- apply the network to the whole test set
predictions = model.predict(x_test)

#-- plot randomly selected noise spectra and compare
plt.subplot(1, 1, 1)
rand_set = np.random.randint( 0, y_test.shape[0] ,(8,) )
for i in rand_set:
    plt.plot( w_train/1e6, y_test[i,:], color='C'+str(i) )
    plt.plot( w_train/1e6, predictions[i], dashes=[2,2,2,2], color='C'+str(i) )
plt.xscale('log')
plt.ylim(1e2,5e4)
plt.yscale('log')
plt.ylabel('Noise Amplitude')
plt.xlabel('Frequency \omega (MHz*2*pi)')
plt.title('Final accuracy = '+str(final_accuracy))
plt.savefig( 'TRAINED_NETWORKS/MODEL_TEST_'+paramchar+'.pdf', format='pdf' )
plt.close()

