
#Import libraries
import numpy as np

import pandas

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

import sys

from sklearn.preprocessing import StandardScaler

#Modding alpha from relu

def relu_alpha(x):
    return K.relu(x, alpha=1.0)

#Making the main function
def run_model(path_x, path_y):

	# Load dataset values
	data_values = pandas.read_csv(path_x, delimiter=",")
	data_values = data_values.values

	# Load labels values
	data_labels = pandas.read_csv(path_y, delimiter=",")
	data_labels = data_labels.values

	# Split into input (X) and output (Y) variables
	X = data_values[:,1:21]
	Y = data_labels[:,1:3]

	#Standard of dataset values for better learning
	scaler = StandardScaler()
	scaler.fit(X)
	X = scaler.transform(X)

	#Making the model
	#1 input layer with 20 inputs
	#1 output layes with 10 entrys and 2 outputs (slop and intercept)
	#Both with relu activation
	model = Sequential()
	model.add(Dense(10, input_shape=(20, )))
	model.add(Activation(relu_alpha))

	model.add(Dense(2, input_shape=(10, )))
	model.add(Activation(relu_alpha))

	#Adam optimizer
	opt = Adam(decay=1e-6)

	#Setting the loss and metrics
	model.compile(optimizer = opt, loss = 'mse', metrics = ['mse','mae'])

	#Fit model with 10 epochs and validation with 0.2 rate from training
	model.fit(X, Y, nb_epoch=10, batch_size=32, validation_split=0.2)

	#Getting prediction
	predictions = model.predict(X)

	#Formating data to export
	output = pandas.DataFrame(predictions, columns = ['slope', 'intercept'])
	output.index.name = 'id'
	output.to_csv('submission.train_100k.csv', sep=',')


if __name__ == "__main__":
    path_x = str(sys.argv[1])
    path_y = str(sys.argv[2])
    run_model(path_x, path_y)