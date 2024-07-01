import numpy as np
import nnfs
nnfs.init()

#@ Dense Layer:
class Dense_Layer:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_L1=0, weight_regularizer_L2=0, bias_regularizer_L1=0, bias_regularizer_L2=0):


        #Initialize Weights and bias:
        self.weights=0.01*np.random.randn(n_inputs, n_neurons)
        self.biases=np.zeros(1, n_neurons)

        # Setting regualrization:
        self.weight_regularizer_L1=weight_regularizer_L1
        self.weight_regularizer_L2=weight_regularizer_L2
        self.bias_regularizer_L1=bias_regularizer_L1
        self.bias_regularizer_L2=bias_regularizer_L2

        #@ Forward Pass:
        def forward(self, inputs, training):
            #remembering the input value:
            self.inputs=inputs
            self.output=np.dot(inputs, self.weights)+self.biases

        #@ Backward Pass:
        def backward(self, dvalues):
            self.dweights=np.dot(self.inputs.T, dvalues)
            self.dbiases=np.sum(dvalues, axis=0, keepdims=True)

            #Regularization:
            if self.weight_regualrizer_L1>0:
                dL1=np.ones_like(self.weights)
                dL1[self.weights<0]=-1
                self.dweights+=self.weight_regularizer_L1*dL1
            
            if self.biases_regularizer_L1>0:
                dL1=np.ones_like(self.biases)
                dL1[self.biases<0]=-1
                self.dbiases+=self.biases_regularizer_L1*dL1

            if self.weight_regularizer_L2>0:
                self.dweights+=2*self.weight_regularizer_L2 * self.weights
            
            if self.bias_regularizer_L2>0:
                self.dbiases+=2*self.bias_regularizer_L2*self.biases

            
            # Gradient on values:
            self.dinputs=np.dot(dvalues, self.weights.T)


#@ Dropout Layer:
class Dropout_Layer:

    def __init__(self, rate):
        self.rate=1-rate

    # Forward Pass:
    def forward(self, inputs, training):
    #saving the inputs:
      self.inputs=inputs

    # In case of training only:
      if not training:
          self.output=inputs.copy()
          return
      
      self.binary_mask=np.random.binomial(1, self.rate, size=inputs.shape)/self.rate   
      self.output=inputs*self.binary_mask   

    # Backward Pass:
    def backward(self, dvalues):
        self.dinputs=dvalues*self.binary_mask       

        
 