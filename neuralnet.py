#Insert code for neural network here
# The NN algorithm is inspired by the structure of the human brain
# and mimics the behaviour of the human brain to solve complex data-driven problems
# NN take input data, train themselves to recognize patterns found in data
# and then predict the output for a new set of similar data

# ANN = artificial neural network --> represents interconnected input
# & output units in which each connection has an associated weight.
# During the learning phase, the network learns by adjusting these weights
# in order to be able to predect the correct class for input data.

# Basically, we assign weights to different values and predict the output from them

# perceptron = NN without any hidden layer (has only input layer & output layer)

# NN executes in 2 steps: 1. feedforward 2. backpropagation
# feedforward NN: we have a set of input features & some random weights (which will be optimized using backward propagation)
# backpropagation: we calculate ERR between predicted output & target output then use an algorithm
# (gradient descent) to update weight values

# Why do we need backpropagation?
# While designing a neural network, first, we need to train a model and assign specific weights to each of those inputs. That weight decides how vital is that feature for our prediction. The higher the weight, the greater the importance. However, initially, we do not know the specific weight required by those inputs. So what we do is, we assign some random weight to our inputs, and our model calculates the error in prediction. Thereafter, we update our weight values and rerun the code (backpropagation). After individual iterations, we can get lower error values and higher accuracy.

# SUMMARY of ANN: 1. take inputs 2. add bias (if required) 3. assign random weights to input features
# 4. run code for training 5. find ERR in prediction 6. update weight by gradient descent algorithm
# 7. repeat training phase with updated weights 8. make predictions

# 6. Sigmoid Function:
# A sigmoid function serves as an activation function in our neural network training. We generally use neural networks for classifications. In binary classification, we have 2 types. However, as we can see, our output value can be any possible number from the equation we used. To solve that problem, we use a sigmoid function. Now for classification, we want our output values to be 0 or 1. So to get values between 0 and 1 we use the sigmoid function. The sigmoid function converts our output values between 0 and 1.

# for any input values, the value of the sigmoid function will always lie between 0 and 