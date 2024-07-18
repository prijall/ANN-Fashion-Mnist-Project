import numpy as np
import os
import cv2
import nnfs
import pickle
import copy

nnfs.init()

#@ Dense Layer:
class Dense_Layer:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_L1=0, weight_regularizer_L2=0, bias_regularizer_L1=0, bias_regularizer_L2=0):

        #Initialize Weights and bias:
        self.weights=0.01*np.random.randn(n_inputs, n_neurons)
        self.biases=np.zeros((1, n_neurons))

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
            if self.weight_regularizer_L1>0:
                dL1=np.ones_like(self.weights)
                dL1[self.weights<0]=-1
                self.dweights+=self.weight_regularizer_L1*dL1
            
            if self.bias_regularizer_L1>0:
                dL1=np.ones_like(self.biases)
                dL1[self.biases<0]=-1
                self.dbiases+=self.bias_regularizer_L1*dL1

            if self.weight_regularizer_L2>0:
                self.dweights+=2*self.weight_regularizer_L2 * self.weights
            
            if self.bias_regularizer_L2>0:
                self.dbiases+=2*self.bias_regularizer_L2*self.biases

            
            # Gradient on values:
            self.dinputs=np.dot(dvalues, self.weights.T)
    
    #@ Retreiving layer parameters:
    def  get_parameters(self):
        return self.weights, self.biases
    
    #@ Setting weights and biases in layer instance:
    def set_parameters(self, parameters):
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)


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
    def forward(self, inputs, training):
        self.output=inputs

#@ ReLU Activation:
class ReLU_Activation:

    #forward pass:
    def forward(self, inputs, training):
        self.inputs=inputs
        self.output=np.maximum(0, inputs)

    #Backward Pass:
    def backward(self, dvalues):
        #since we intend to modify the original values, we need to make a copy:
        self.dinputs=dvalues.copy()
        self.dinputs[self.inputs<=0]=0

    def prediction(self, outputs):
        return outputs


#@ Softmax Activaiton:
class Softmax_activation:

    #forward pass:
    def forward(self, inputs, training):
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
    def predictions(self, outputs):
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
    def __init__(self, learning_rate=0.01, decay=0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.epsilon=epsilon
        self.beta_1=beta_1
        self.beta_2=beta_2
        self.iterations=0

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
        layer.weight_momentums=self.beta_1 *layer.weight_momentums + (1 - self.beta_1)*layer.weights
        layer.bias_momentums=self.beta_1 *layer.bias_momentums + (1 - self.beta_1)*layer.biases

        weight_momentums_corrected=layer.weight_momentums / (1- self.beta_1 ** (self.iterations+1))
        bias_momentums_corrected=layer.bias_momentums / (1- self.beta_1 ** (self.iterations+1))

        layer.weight_cache=self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.weights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache +  (1 - self.beta_2) * layer.biases**2

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
            if layer.weight_regularizer_L1>0:
                regularization_loss+=layer.regularizer_L1*np.sum(np.abs(layer.weights))
            
            if layer.weight_regularizer_L2>0:
                regularization_loss+=layer.weight_regularizer_L2 * np.sum(layer.weights * layer.weights)

            if layer.bias_regularizer_L1>0:
                regularization_loss+=layer.bias_regularizer_L1 * np.sum(np.abs(layer.biases))
            
            if layer.bias_regularizer_L2>0:
                regularization_loss+=layer.bias_regularizer_L2* np.sum(layer.biases*np.biases)


            return regularization_loss
    
    #for remembering trainable params:
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers=trainable_layers


    #for calculation:
    def calculate(self, output, y, *, include_regularization=False):
        sample_losses=self.forward(output, y)
        data_loss=np.mean(sample_losses)

        #adding accumulated sum of losses and sample count:
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count+=len(sample_losses)

        if not include_regularization:
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


# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

   def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
          correct_confidences = y_pred_clipped[range(samples), y_true]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
         correct_confidences = np.sum(y_pred_clipped * y_true,axis=1)
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
   
        # Backward pass
   def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
         y_true = np.eye(labels)[y_true]
        # Calculate gradient
         self.dinputs = -y_true / dvalues
        # Normalize gradient
         self.dinputs = self.dinputs / samples
     
class Activation_Softmax_Loss_CategoricalCrossentropy():
        # Backward pass
        def backward(self, dvalues, y_true):
        # Number of samples
            samples = len(dvalues)
            # If labels are one-hot encoded,
            # turn them into discrete values
            if len(y_true.shape) == 2:
                y_true = np.argmax(y_true, axis=1)
            # Copy so we can safely modify
            self.dinputs = dvalues.copy()
            # Calculate gradient
            self.dinputs[range(samples), y_true] -= 1
            # Normalize gradient
            self.dinputs = self.dinputs / samples


            # Binary cross-entropy loss
class Loss_BinaryCrossentropy(Loss):
        # Forward pass
        def forward(self, y_pred, y_true):
            # Clip data to prevent division by 0
            # Clip both sides to not drag mean towards any value
            y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
            # Calculate sample-wise loss
            sample_losses = -(y_true * np.log(y_pred_clipped) +
            (1 - y_true) * np.log(1 - y_pred_clipped))
            sample_losses = np.mean(sample_losses, axis=-1)
            # Return losses
            return sample_losses
            # Backward pass
        def backward(self, dvalues, y_true):
            # Number of samples
             samples = len(dvalues)
            # Number of outputs in every sample
            # We'll use the first sample to count them
             outputs = len(dvalues[0])
    
            # Clip data to prevent division by 0
            # Clip both sides to not drag mean towards any value
             clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
            # Calculate gradient
             self.dinputs = -(y_true / clipped_dvalues -
            (1 - y_true) / (1 - clipped_dvalues)) / outputs
            # Normalize gradient
             self.dinputs = self.dinputs / samples


#@ for accuracy:
class Accuracy:
    def calculate(self, predictions, y):
        comparisons=self.compare(predictions, y)
        accuracy=np.mean(comparisons)

        #adding acculumated sum of matching values and sample count:
        self.accumulated_sum+= np.sum(comparisons)
        self.accumulated_count+= len(comparisons)

        return accuracy
    
    #for accumulated accuracy:
    def calculate_accumulated(self):
        accuracy=self.accumulated_sum / self.accumulated_count
        return accuracy
    
    #reseting variable for accumulated accuracy:
    def new_pass(self):
        self.accumulated_sum=0 
        self.accumulated_count=0


#@ Accuracy for classification model:
class Accuracy_Categorical(Accuracy):
    def init(self, y):
        pass

    def compare(self, predictions, y):
        if len(y.shape)==2:
            y=np.argmax(y, axis=1)
        return predictions==y
    
#@ Accuracy for regression model:
class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.precision=None

    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision=np.std(y)/ 250

    def compare(self, predictions, y):
        return np.absolute(predictions-y)<self.precision



# Model class
class Model:

    def __init__(self):
        # Create a list of network objects
        self.layers = []
        # Softmax classifier's output object
        self.softmax_classifier_output = None

    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)


    # Set loss, optimizer and accuracy
    def set(self, *, loss=None, optimizer=None, accuracy=None):

        if loss is not None:
            self.loss = loss

        if optimizer is not None:
            self.optimizer = optimizer

        if accuracy is not None:
            self.accuracy = accuracy

    # Finalize the model
    def finalize(self):

        # Create and set the input layer
        self.input_layer = Layer_Input()

        # Count all the objects
        layer_count = len(self.layers)

        # Initialize a list containing trainable layers:
        self.trainable_layers = []

        # Iterate the objects
        for i in range(layer_count):

            # If it's the first layer,
            # the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            # All layers except for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            # The last layer - the next object is the loss
            # Also let's save aside the reference to the last object
            # whose output is the model's output
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]


            # If layer contains an attribute called "weights",
            # it's a trainable layer -
            # add it to the list of trainable layers
            # We don't need to check for biases -
            # checking for weights is enough
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        # Update loss object with trainable layers
        if self.loss is not None:
            self.loss.remember_trainable_layers(
                self.trainable_layers
            )

        # If output activation is Softmax and
        # loss function is Categorical Cross-Entropy
        # create an object of combined activation
        # and loss function containing
        # faster gradient calculation
        if isinstance(self.layers[-1], Activation_Softmax) and \
           isinstance(self.loss, Loss_CategoricalCrossentropy):
            # Create an object of combined activation
            # and loss functions
            self.softmax_classifier_output = \
                Activation_Softmax_Loss_CategoricalCrossentropy()

    # Train the model
    def train(self, X, y, *, epochs=1, batch_size=None,
              print_every=1, validation_data=None):

        # Initialize accuracy object
        self.accuracy.init(y)

        # Default value if batch size is not being set
        train_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size
            # Dividing rounds down. If there are some remaining
            # data but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if train_steps * batch_size < len(X):
                train_steps += 1


        # Main training loop
        for epoch in range(1, epochs+1):

            # Print epoch number
            print(f'epoch: {epoch}')

            # Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()

            # Iterate over steps
            for step in range(train_steps):

                # If batch size is not set -
                # train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y

                # Otherwise slice a batch
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                # Perform the forward pass
                output = self.forward(batch_X, training=True)

                # Calculate loss
                data_loss, regularization_loss = \
                    self.loss.calculate(output, batch_y,
                                        include_regularization=True)
                loss = data_loss + regularization_loss

                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(
                                  output)
                accuracy = self.accuracy.calculate(predictions,
                                                   batch_y)

                # Perform backward pass
                self.backward(output, batch_y)

                # Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()


                # Print a summary
                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {regularization_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate}')

            # Get and print epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = \
                self.loss.calculate_accumulated(
                    include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}')

            # If there is the validation data
            if validation_data is not None:

                # Evaluate the model:
                self.evaluate(*validation_data,
                              batch_size=batch_size)

    # Evaluates the model using passed-in dataset
    def evaluate(self, X_val, y_val, *, batch_size=None):

        # Default value if batch size is not being set
        validation_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            # Dividing rounds down. If there are some remaining
            # data but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        # Reset accumulated values in loss
        # and accuracy objects
        self.loss.new_pass()
        self.accuracy.new_pass()


        # Iterate over steps
        for step in range(validation_steps):

            # If batch size is not set -
            # train using one step and full dataset
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val

            # Otherwise slice a batch
            else:
                batch_X = X_val[
                    step*batch_size:(step+1)*batch_size
                ]
                batch_y = y_val[
                    step*batch_size:(step+1)*batch_size
                ]

            # Perform the forward pass
            output = self.forward(batch_X, training=False)

            # Calculate the loss
            self.loss.calculate(output, batch_y)

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(
                              output)
            self.accuracy.calculate(predictions, batch_y)

        # Get and print validation loss and accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        # Print a summary
        print(f'validation, ' +
              f'acc: {validation_accuracy:.3f}, ' +
              f'loss: {validation_loss:.3f}')

    # Predicts on the samples
    def predict(self, X, *, batch_size=None):

        # Default value if batch size is not being set
        prediction_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            prediction_steps = len(X) // batch_size

            # Dividing rounds down. If there are some remaining
            # data but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        # Model outputs
        output = []

        # Iterate over steps
        for step in range(prediction_steps):

            # If batch size is not set -
            # train using one step and full dataset
            if batch_size is None:
                batch_X = X

            # Otherwise slice a batch
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]

            # Perform the forward pass
            batch_output = self.forward(batch_X, training=False)

            # Append batch prediction to the list of predictions
            output.append(batch_output)

        # Stack and return results
        return np.vstack(output)

    # Performs forward pass
    def forward(self, X, training):

        # Call forward method on the input layer
        # this will set the output property that
        # the first layer in "prev" object is expecting
        self.input_layer.forward(X, training)

        # Call forward method of every object in a chain
        # Pass output of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # "layer" is now the last object from the list,
        # return its output
        return layer.output


    # Performs backward pass
    def backward(self, output, y):

        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # First call backward method
            # on the combined activation/loss
            # this will set dinputs property
            self.softmax_classifier_output.backward(output, y)

            # Since we'll not call backward method of the last layer
            # which is Softmax activation
            # as we used combined activation/loss
            # object, let's set dinputs in this object
            self.layers[-1].dinputs = \
                self.softmax_classifier_output.dinputs

            # Call backward method going through
            # all the objects but last
            # in reversed order passing dinputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        # First call backward method on the loss
        # this will set dinputs property that the last
        # layer will try to access shortly
        self.loss.backward(output, y)

        # Call backward method going through all the objects
        # in reversed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    # Retrieves and returns parameters of trainable layers
    def get_parameters(self):

        # Create a list for parameters
        parameters = []

        # Iterable trainable layers and get their parameters
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        # Return a list
        return parameters


    # Updates the model with new parameters
    def set_parameters(self, parameters):

        # Iterate over the parameters and layers
        # and update each layers with each set of the parameters
        for parameter_set, layer in zip(parameters,
                                        self.trainable_layers):
            layer.set_parameters(*parameter_set)

    # Saves the parameters to a file
    def save_parameters(self, path):

        # Open a file in the binary-write mode
        # and save parameters into it
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    # Loads the weights and updates a model instance with them
    def load_parameters(self, path):

        # Open file in the binary-read mode,
        # load weights and update trainable layers
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    # Saves the model
    def save(self, path):

        # Make a deep copy of current model instance
        model = copy.deepcopy(self)

        # Reset accumulated values in loss and accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()

        # Remove data from the input layer
        # and gradients from the loss object
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        # For each layer remove inputs, output and dinputs properties
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs',
                             'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        # Open a file in the binary-write mode and save the model
        with open(path, 'wb') as f:
            pickle.dump(model, f)


    # Loads and returns a model
    @staticmethod
    def load(path):
        try: 
        # Open file in the binary-read mode, load a model
          with open(path, 'rb') as f:
            model = pickle.load(f)
        except FileNotFoundError:
            print(f"File not found: {path}")
        except PermissionError:
            print(f"Permission denied: {path}")
        except Exception as e:
            print(f"An error occurred: {e}")
path = 'fashion_mnist_images'

# Check if the file exists
if os.path.exists(path):
    print(f"File exists: {path}")
    # Change the file permissions to allow reading (if necessary)
    os.chmod(path, 0o644)

    # Now try to load the model
    model = Model.load(path)
else:
    print(f"The file '{path}' does not exist.")

   


# Loads a MNIST dataset
def load_mnist_dataset(dataset, path):

    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # Create lists for samples and labels
    X = []
    y = []

    # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(
                        os.path.join(path, dataset, label, file),
                        cv2.IMREAD_UNCHANGED)

            # And append it and a label to the lists
            X.append(image)
            y.append(label)

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')


# MNIST dataset (train + test)
def create_data_mnist(path):

    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    # And return all the data
    return X, y, X_test, y_test



# Label index to label name relation
fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# Read an image
image_data = cv2.imread('pants.png', cv2.IMREAD_GRAYSCALE)

# Resize to the same size as Fashion MNIST images
image_data = cv2.resize(image_data, (28, 28))

# Invert image colors
image_data = 255 - image_data

# Reshape and scale pixel data
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

# Load the model
model = Model.load('fashion_mnist_images')

# Predict on the image
confidences = model.predict(image_data)

# Get prediction instead of confidence levels
predictions = model.output_layer_activation.predictions(confidences)

# Get label name from label index
prediction = fashion_mnist_labels[predictions[0]]

print(prediction)