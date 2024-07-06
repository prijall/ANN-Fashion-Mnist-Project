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

        
 #@ Input Layer:
class Input_Layer:
    #forward pass:
    def forward(self, inputs):
        self.output=inputs

#@ ReLU Activation:
class ReLU_Activation:

    #forward pass:
    def forward(self, inputs):
        self.inputs=inputs
        self.output=np.maximum(0, inputs)

    #Backward Pass:
    def backward(self, dvalues):
        #since we intend to modify the original values, we need to make a copy:
        self.dinputs=dvalues.copy()
        self.dinputs[self.inputs<=0]=0


#@ Softmax Activaiton:
class Softmax_activation:

    #forward pass:
    def forward(self, inputs):
        self.inputs= inputs
        exp_values=np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        probabilities=exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output=probabilities

     #Backward Pass:
    def backward(self, dvalues):
        self.dinputs=np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output=single_output.reshape(-1, 1)
            jacobian_matrix=np.diagflat(single_output)- np.dot(single_output, single_output.T)
            self.dinputs[index]=np.dot(jacobian_matrix, single_dvalues)
     
      #prediction for output:
    def prediction(self, outputs):
        return np.argmax(outputs, axis=1)

#@ Sigmoid Activation:
class Sigmoid_Activation:

    #forward pass:
    def forward(self, inputs):
        self.inputs=inputs
        self.output=1/(1+np.exp(-inputs))

    #Backward Pass:
    def backward(self, dvalues):
        self.dinputs=dvalues*(1-self.output)*self.output

    #prediction for output:
    def prediction(self, outputs):
        return (outputs>0.5)*1
    

#@ Linear Activation:
class Linear_Activation:
    #forward pass:
    def forward(self, inputs):
        self.inputs=inputs
        self.output=inputs

    # Backward Pass:
    def backward(self, dvalues):
        self.dinputs=dvalues.copy()

    #prediction for output:
    def predictions(self, outputs):
        return outputs
    
#@ Adam optimizer:
class Adam_Optimizer:
    def __init(self, learning_rate, decay, epsilon, beta_1, beta_2):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.epsilon=epsilon
        self.beta_1=beta_1
        self.beta_2=beta_2

    #call once before any parameter update:
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate =self.learning_rate * (1 / (1 + self.decay*self.iterations))
    
    # Update Parameters:
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums=np.zeros_like(layer.weights)
            layer.weight_cache=np.zeros_like(layer.weights)
            layer.bias_momentums=np.zeros_like(layer.biases)
            layer.bias_cache=np.zeros_like(layer.biases)
        
        #Update momentum with current gradients:
        layer.weight_momentums=self.beta_1 *layer.weight_momentum + (1 - self.beta_1)*layer.dweights
        layer.bias_momentums=self.beta_1 *layer.bias_momentum + (1 - self.beta_1)*layer.dbiases

        weight_momentums_corrected=layer.weight_momentums / (1- self.beta_1 ** (self.iterations+1))
        bias_momentums_corrected=layer.bias_momentums / (1- self.beta_1 ** (self.iterations+1))

        layer.weight_cache=self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache +  (1 - self.beta_2) * layer.dbiases**2

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
    
    # Call once after parameter update:
    def post_update_params(self):
        self.iterations+=1


#@ Loss:
class Loss:
    def regularization_loss(self):
        regularization_loss=0 #by default
        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1>0:
                regularization_loss+=layer.regularizer_l1*np.sum(np.abs(layer.weights))
            
            if layer.weight_regularizer_l2>0:
                regularization_loss+=layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

            if layer.bias_regularizer_l1>0:
                regularization_loss+=layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            
            if layer.bias_regularizer_l2>0:
                regularization_loss+=layer.bias_regularizer_l2* np.sum(layer.biases*np.biases)


            return regularization_loss
    
    #for remembering trainable params:
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers=trainable_layers


    #for calculation:
    def calculate(self, output, y, *, include_regualarization=False):
        sample_losses=self.forward(output, y)
        data_loss=np.mean(sample_losses)

        #adding accumulated sum of losses and sample count:
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count+=len(sample_losses)

        if not include_regualarization:
            return data_loss
        
        return data_loss, self.regularization_loss()
    
    #for calculating accumulated loss:
    def calculate_accumulated(self, *, include_regularization=False):
        data_loss=self.accumulated_sum/self.accumulated_count

        if not include_regularization:
            return data_loss
        
    
        return data_loss, self.regularization_loss()
    

    #For reseting variables for accumulated loss:
    def new_pass(self):
        self.accumulated_sum=0
        self.accumulated_count=0