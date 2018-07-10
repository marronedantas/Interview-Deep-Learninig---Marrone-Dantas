import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load dataset
data_values = pandas.read_csv("train_100k.csv", delimiter=",")
data_values = data_values.values

data_labels = pandas.read_csv("train_100k.truth.csv", delimiter=",")
data_labels = data_labels.values

# split into input (X) and output (Y) variables
X = data_values[:,1:21]
Y = data_labels[:,1:2]

#building model
model = Sequential()
model.add(Dense(20, input_dim=20, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#making estimator
estimator = KerasRegressor(model, epochs=100, batch_size=5, verbose=0)

#output
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
