from tensorflow.keras import models, layers

def get_model( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size, net_type ):

	if net_type==1:
		model = get_model_1( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size )
	elif net_type==2:
		model = get_model_2( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size )
	elif net_type==3:
		model = get_model_3( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size )
	elif net_type==4:
		model = get_model_4( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size )
	elif net_type==5:
		model = get_model_5( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size )
	elif net_type==6:
		model = get_model_6( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size )
	elif net_type==7:
		model = get_model_7( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size )
	elif net_type==8:
		model = get_model_8( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size )
	elif net_type==9:
		model = get_model_9( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size )
	
	return model

def get_model_1( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):

	model = models.Sequential()
	model.add( layers.Input( shape=(xtrain_size, 1) ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb//2, kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb//4,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )

	model.add( layers.Conv1D( filter_nb//4,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )

	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.Conv1D( 1, kernel_size, activation="relu", padding='same' ) )
	model.add( layers.Flatten() )
	model.add( layers.Dropout( dropout_rate ) )
	model.add( layers.Dense(501, activation='linear') )

	return model

def get_model_2( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):
	
	model = models.Sequential()
	model.add( layers.Input( shape=(xtrain_size, 1) ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb//2, kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb//4,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb//8,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )

	model.add( layers.Conv1D( filter_nb//8,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb//4,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )

	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.Conv1D( 1, kernel_size, activation="relu", padding='same' ) )
	model.add( layers.Flatten() )
	model.add( layers.Dropout( dropout_rate ) )
	model.add( layers.Dense(501, activation='linear') )

	return model

def get_model_3( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):
	
	model = models.Sequential()
	model.add( layers.Input( shape=(xtrain_size, 1) ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb, kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )

	model.add( layers.Conv1D( filter_nb*2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb*4,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb*8,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.Conv1D( 1, kernel_size, activation="relu", padding='same' ) )
	model.add( layers.Flatten() )
	model.add( layers.Dropout( dropout_rate ) )
	model.add( layers.Dense(501, activation='linear') )

	return model

def get_model_4( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):
	
	#-- assessement: inefficient
	model = models.Sequential()
	model.add( layers.Input( shape=(xtrain_size, 1) ) )
	model.add( layers.Conv1D(filter_nb//8,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb//4, kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb//2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )

	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.Conv1D( 1, kernel_size, activation="relu", padding='same' ) )
	model.add( layers.Flatten() )
	model.add( layers.Dropout( dropout_rate ) )
	model.add( layers.Dense(501, activation='linear') )

	return model

def get_model_5( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):
	
	model = models.Sequential()
	model.add( layers.Input( shape=(xtrain_size, 1) ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb, kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )

	model.add( layers.Conv1D( filter_nb*2,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb*4,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb*8,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.Conv1D( 1, kernel_size, activation="relu", padding='same' ) )
	model.add( layers.Flatten() )
	model.add( layers.Dropout( dropout_rate ) )
	model.add( layers.Dense(501, activation='linear') )

	return model

def get_model_6( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):
	
	model = models.Sequential()
	model.add( layers.Input( shape=(xtrain_size, 1) ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb, kernel_size//2,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb,kernel_size//4,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb,kernel_size//8,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb,kernel_size//16,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D( filter_nb,kernel_size//16,activation="relu", padding='same' ) )
	model.add( layers.Conv1D( 1, kernel_size, activation="relu", padding='same' ) )
	model.add( layers.Flatten() )
	model.add( layers.Dropout( dropout_rate ) )
	model.add( layers.Dense(501, activation='linear') )

	return model

def get_model_7( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):
	
	model = models.Sequential()
	model.add( layers.Input( shape=(xtrain_size, 1) ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb, kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb,kernel_size//4,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb,kernel_size//4,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D( filter_nb,kernel_size//4,activation="relu", padding='same' ) )
	model.add( layers.Conv1D( 1, kernel_size, activation="relu", padding='same' ) )
	model.add( layers.Flatten() )
	model.add( layers.Dropout( dropout_rate ) )
	model.add( layers.Dense(501, activation='linear') )

	return model

def get_model_8( filter_nb, kernel_size, pool_size, dropout_rate, xtrain_size ):
	
	model = models.Sequential()
	model.add( layers.Input( shape=(xtrain_size, 1) ) )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb, kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )
	model.add( layers.Conv1D(filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.MaxPooling1D( pool_size=pool_size, padding="same") )

	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.UpSampling1D( size=pool_size ) )
	model.add( layers.Conv1D( filter_nb,kernel_size,activation="relu", padding='same' ) )
	model.add( layers.Conv1D( 1, kernel_size, activation="relu", padding='same' ) )
	model.add( layers.Flatten() )
	model.add( layers.Dropout( dropout_rate ) )
	model.add( layers.Dense(501, activation='linear') )

	return model
