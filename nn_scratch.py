import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nn_functions import *
from keras.datasets import mnist 


def model(X,Y,layer_dims,learning_rate=0.01,epochs=100,print_cost=False,optimizer="gd"):

	np.random.seed(1)
	parameters=init_weights(layer_dims)

	if(optimizer=="momentum"):
		v=init_velocity(parameters)

	if(optimizer=="adam"):
		v,s=init_adam(parameters)
	
	costs=[]
	

	for i in range(0,epochs):
		preds,caches=forward_propagation(X,parameters)
		cost_=cost(preds,Y)
		grads=back_propagation(preds,Y,caches)
		
		if(optimizer=="momentum"):
			parameters,v=update_with_momentum(parameters, grads, v, learning_rate = learning_rate)

		elif(optimizer=="adam"):
			parameters,v,s=update_with_adam(parameters, grads, v, s,learning_rate = learning_rate)

		elif(optimizer=="gd"):
			parameters=update_parameters(parameters,grads,learning_rate)

		if print_cost and i%100==0:
			print("Cost after iteration %i : %f" %(i,cost_))
			costs.append(cost_)

	return parameters,costs

def visualize(costs,learning_rate=0.01):

	plt.plot(np.squeeze(costs))
	plt.xlabel('Cost')
	plt.ylabel('No.of Iterations')
	plt.title("Learning Rate ="+str(learning_rate))
	plt.show()
	
(x_train, y_train), (x_test, y_test) = mnist.load_data()


train_x_flatten = x_train.reshape(x_train.shape[0], -1).T   
test_x_flatten = x_test.reshape(x_test.shape[0], -1).T

train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

train_y=one_hot_encoding(y_train,10).T

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(train_y.shape))

layers_dims = [784,10]

parameters,costs = model(train_x, train_y, layers_dims, epochs = 2500, print_cost = True,optimizer='gd')
visualize(costs)
