# Interview Deep Learninig - Marrone-Dantas

## The Task
The task is to create a neural network which takes a set of 10 points as inputs, and outputs slope and the y-intercept of the best-fitting line for the given points. The points are noisy, i.e. they won't fit perfectly on a line, so the net must figure out the best-fit line.

## Approach 
Fot the aproacha was made a simple Neural Network with composed was a network with a input layer with 20 values (X1,Y1,...(X9,Y9).
The intermediare layer with 10 inputs, and finaly the output layer with 2 outputs.
```
model = Sequential()
model.add(Dense(10, input_shape=(20, )))
model.add(Activation(relu_alpha))
model.add(Dense(2, input_shape=(10, )))
model.add(Activation(relu_alpha))
```

## Observations

1. The dataser pass by standardization for better learning
```
scaler = StandardScaler()
	scaler.fit(X)
	X = scaler.transform(X)
```
2. Was used the Relu activation with modification of alpha, since the prediction have a since the outputs have a large variation.
```
def relu_alpha(x):
    return K.relu(x, alpha=1.0)
```
3. The adaptive option (ADAM optimizer) demonstrated better perfomance than other approaches with SGD
```
opt = Adam(decay=1e-6)
```
4. The maximum convergence occurs with a few times (10), an indication that this occurs in the network is the result of validation (0.2% from training) to follow the training error.
5. The rest of the code are just the exportation.

## Consideration

1. The proposed problem is a regression, so it demands a different approach, as softmax our outhers activations are not applicable in the classification model.
2. The distribution of the code may deserve a treatment, since the components of the subtasts can be arranged more dynamically.
3. The difficulty in this network was with the intercept, it causes a huge noise, difucultando the learning of the network.
4. Reliably the base is not huge and other approaches could be better applied and simpler.
5. Indications that with the addition of a few more hidden cams and a bigger workout the result can be improved.

## Usage

The following is a description of how to execute the code and the necessary components

### Libraries

1. Keras
2. Tensorflow
3. Scikit-learn

### How to run

The first parameter is the base and second the desired values of output
```
$ python nn.py train_100k.csv train_100k.truth.csv
```
The code generates the outout file "submission.train_100k.csv" with the predictions.
