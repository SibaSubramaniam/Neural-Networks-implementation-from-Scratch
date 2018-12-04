import numpy as np

def init_weights(layer_dims):

	np.random.seed(3)

	parameters={}
	L=len(layer_dims)

	for l in range(1,L):
		parameters['W'+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])*1.414 /np.sqrt(layer_dims[l-1])
		parameters['b'+str(l)]=np.zeros((layer_dims[l],1))

	return parameters

def sigmoid(z):
	
	a=1/(1+np.exp(-z))
	return a,z

def relu(z):

	a=np.where(z>0,z,0)
	return a,z

def linear_forward(x,W,b):
	
	z=np.dot(W,x)+b
	cache=(x,W,b)

	return z,cache

def linear_activation_forward(x,W,b,activation):

	z,linear_cache=linear_forward(x,W,b)

	if(activation=='sigmoid'):
		a,activation_cache=sigmoid(z)

	if(activation=='relu'):
		a,activation_cache=relu(z)

	cache=(linear_cache,activation_cache)

	return a,cache

def forward_propagation(X,parameters):

	l=len(parameters)//2
	A=X
	caches=[]

	for i in range(1,l):
		temp=A
		A,cache=linear_activation_forward(temp,parameters['W'+str(i)],parameters['b'+str(i)],"relu")
		caches.append(cache)

	# i=i+1
	AL,cache=linear_activation_forward(A,parameters['W'+str(l)],parameters['b'+str(l)],"sigmoid")
	caches.append(cache)

	return AL,caches

def cost(pred,y):
	
	m=y.shape[1]
	cost=np.sum(y*np.log(pred)+(1-y)*np.log(1-pred))*-1/m
	cost=np.squeeze(cost)

	return cost

def sigmoid_backward(da,cache):

	z=cache
	s=1/(1+np.exp(-z))
	dz=da*s*(1-s)

	return dz

def relu_backward(da,cache):

	z=cache
	dz=np.array(da,copy=True)
	dz[z<=0]=0

	return dz

def linear_backward(dz,cache):

	A,W,b=cache
	m=A.shape[1]

	dW=1./m*np.dot(dz,A.T)
	db=1./m*np.sum(dz,axis=1,keepdims=True)
	da=np.dot(W.T,dz)

	return da,dW,db

def linear_activation_backward(da,cache,activation):

	linear_cache,activation_cache=cache

	if(activation=="relu"):
		dz=relu_backward(da,activation_cache)
		da,dW,db=linear_backward(dz,linear_cache)

	if(activation=="sigmoid"):
		dz=sigmoid_backward(da,activation_cache)
		da,dW,db=linear_backward(dz,linear_cache)

	return da,dW,db

def back_propagation(preds,y,caches):
	
	grads={}
	L=len(caches)
	m=preds.shape[1]

	da=-(np.divide(y,preds)-np.divide(1-y,1-preds))
	current_cache=caches[L-1]
	grads["da"+str(L-1)],grads["dW"+str(L)],grads["db"+str(L)]=linear_activation_backward(da,current_cache,"sigmoid")

	for l in reversed(range(L-1)):

		current_cache=caches[l]
		grads["da"+str(l)],grads["dW"+str(l+1)],grads["db"+str(l+1)]=linear_activation_backward(grads["da"+str(l+1)],current_cache,"relu")

	return grads

def update_parameters(parameters,grads,learning_rate):

	L=len(parameters)//2

	for l in range(L):
		parameters["W"+str(l+1)]=parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]
		parameters["b"+str(l+1)]=parameters["b"+str(l+1)]-learning_rate*grads["db"+str(l+1)]

	return parameters

def init_velocity(parameters):

	L=len(parameters)//2
	v={}
	
	for i in range(L):
		v['dW'+str(i+1)]=np.zeros(parameters["W"+str(i+1)].shape)
		v['db'+str(i+1)]=np.zeros(parameters["b"+str(i+1)].shape)

	return v

def init_adam(parameters):

	L=len(parameters)//2
	v={}
	s={}
	for i in range(L):
		v['dW'+str(i+1)]=np.zeros(parameters["W"+str(i+1)].shape)
		v['db'+str(i+1)]=np.zeros(parameters["b"+str(i+1)].shape)
		s['dW'+str(i+1)]=np.zeros(parameters["W"+str(i+1)].shape)
		s['db'+str(i+1)]=np.zeros(parameters["b"+str(i+1)].shape)

	return v,s


def update_with_momentum(parameters,grads,v,learning_rate,beta=0.9):

	L=len(parameters)//2
	v_corrected={}

	for i in range(L):
		v['dW'+str(i+1)]=beta*v['dW'+str(i+1)] + (1-beta)*grads['dW'+str(i+1)]
		v['db'+str(i+1)]=beta*v['db'+str(i+1)] + (1-beta)*grads['db'+str(i+1)]

		v_corrected['dW'+str(i+1)]=v['dW'+str(i+1)]/(1-beta)**(i+1)
		v_corrected['db'+str(i+1)]=v['db'+str(i+1)]/(1-beta)**(i+1)

		parameters['W'+str(i+1)]=parameters['W'+str(i+1)] - learning_rate*(v_corrected['dW'+str(i+1)])
		parameters['b'+str(i+1)]=parameters['b'+str(i+1)] - learning_rate*(v_corrected['db'+str(i+1)])

	return parameters,v


def update_with_adam(parameters, grads, v, s,learning_rate = 0.01,beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):

	L=len(parameters)//2
	v_corrected={}
	s_corrected={}

	for i in range(L):
		v['dW'+str(i+1)]=beta1*v['dW'+str(i+1)] + (1-beta1)*grads['dW'+str(i+1)]
		v['db'+str(i+1)]=beta1*v['db'+str(i+1)] + (1-beta1)*grads['db'+str(i+1)]

		v_corrected['dW'+str(i+1)]=v['dW'+str(i+1)]/(1-beta1)**(i+1)
		v_corrected['db'+str(i+1)]=v['db'+str(i+1)]/(1-beta1)**(i+1)

		s['dW'+str(i+1)]=beta1*s['dW'+str(i+1)] + (1-beta2)*(grads['dW'+str(i+1)]**2)
		s['db'+str(i+1)]=beta1*s['db'+str(i+1)] + (1-beta2)*(grads['db'+str(i+1)]**2)

		s_corrected['dW'+str(i+1)]=s['dW'+str(i+1)]/(1-beta2)**(i+1)
		s_corrected['db'+str(i+1)]=s['db'+str(i+1)]/(1-beta2)**(i+1)


		parameters['W'+str(i+1)]=parameters['W'+str(i+1)] - learning_rate*(v_corrected['dW'+str(i+1)]/
																	np.sqrt(s_corrected['dW'+str(i+1)]+epsilon))
		parameters['b'+str(i+1)]=parameters['b'+str(i+1)] - learning_rate*(v_corrected['db'+str(i+1)]/
																	np.sqrt(s_corrected['db'+str(i+1)]+epsilon))

	return parameters,v,s

def random_minibatches(X,Y,mini_batch_size=64):

	m=X.shape[1]
	mini_batches=[]

	permutation=list(np.random.permutation(m))
	shuffled_x=X[:,permutation]
	shuffled_y=Y[:,permutation]

	mini_batches_n=int(m/mini_batch_size)

	for k in range(0,mini_batches_n):
		mini_batch_x=shuffled_x[:,k*mini_batch_size:(k+1)*mini_batch_size]
		mini_batch_y=shuffled_y[:,k*mini_batch_size:(k+1)*mini_batch_size]

		mini_batch=(mini_batch_x,mini_batch_y)
		mini_batches.append(mini_batch)

	if m%mini_batch_size !=0:
		mini_batch_x=shuffled_x[:,(k+1)*mini_batch_size:]
		mini_batch_y=shuffled_y[:,(k+1)*mini_batch_size:]
		
		mini_batch=(mini_batch_x,mini_batch_y)
		mini_batches.append(mini_batch)

	return mini_batches

def one_hot_encoding(y,size):

	encoded=[]

	for i in y:
		temp=np.zeros(size)
		temp[i]=1
		encoded.append(temp)

	encoded=np.array(encoded)

	return encoded
