import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
from tensorflow.keras import Model
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import linalg as LA
from matplotlib.pyplot import figure
from .utilsProject import *

"""
Class for plain networks. Input shape, depth of the network,number of neurons in each layer, non-linear activation function
type, method of weight initialiation and the number of outputs are passed as argument.
"""
class PlainModel:
    def __init__(self,input_shape,depth,NeuronWidth,activation,initializer,numberOutput):
        self.depth = depth #number of fully connected layers in the network
        self.input_shape=input_shape 
        self.activation=activation #non-linear activation function type passed as string argument.
        self.accArray=[] #accuracy array, accuracy values of each epoch is appended to this array if custom train function is used. 
        self.lossArray=[] #loss array, loss values of each epoch is appended to this array if custom train function is used. 
        self.initializer=initializer #weight initializer
        self.model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=input_shape)]) #Flatten layer is done first for the succesive fully connected layers of the network. 
        for i in range(self.depth): #Dense layers are appended in the for loop using the depth paramater. Initializer and the activation types are used here. 
          self.model.add(tf.keras.layers.Dense(NeuronWidth,activation=activation,kernel_initializer=initializer))
        self.model.add(tf.keras.layers.Dense(numberOutput,kernel_initializer=initializer)) #Final dense layer for classification.   

    #Initializing the loss function and the metrics of the performance of the network for training and the test data.
    #This function is used before custom train function. 
    def init_loss(self,):
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    #Initializing the optimizer for the network. Learning rate and the momenntum values are passed as argument.
    #This function is used before custom train function. 
    def init_optimizer(self,lr,momentum):
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)

    def test_step(self, images, labels):
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()
        predictions = self.model.predict(ds_test)
        true_categories = tf.concat([y for x, y in ds_test], axis=0)
        t_loss = self.loss_function(true_categories, predictions)
        self.test_loss(t_loss)
        self.test_accuracy(true_categories,predictions)

    #This function is for plotting the training accuracy values for each epoch. The values appended to the accArray
    #is visualized. This function is used if custom train function is used.
    def plotTrainingAccuracy(self,):
      temp=np.array(self.accArray)
      plt.plot([*range(1,temp.shape[0]+1,1)],temp)
      plt.ylabel("training accuracy")
      plt.xlabel("number of epochs")
      plt.ylim([0,1])
      
    #This function is for plotting the training loss values for each epoch. The values appended to the lossArray
    #is visualized. This function is used if custom train function is used.
    def plotTrainingLoss(self,):
      temp=np.array(self.lossArray)
      plt.plot([*range(1,temp.shape[0]+1,1)],temp)
      plt.ylabel("training loss")
      plt.xlabel("number of epochs")

    """
    This function is used for custom training. However, this function is used only for debugging since it is very slow 
    compared to TF fit function. Model can be using this function. Norm of the gradients at each step appended to the normGradEpoch
    for debugging and visualization purposes. The training dataset and the number of arguments are passed as an argument.
    """
    def train(self,numberofEpochs,ds_train):
      normGradEpoch=[] #creating empty list for the norm of the gradients.
      self.train_loss.reset_states()
      self.train_accuracy.reset_states()
      for i in range(numberofEpochs): 
        counter=1
        epochGrad=np.zeros(((self.depth*2)+2,)) #creating empty array for the gradients at each step.
        for step,(img_batch,lbl_batch) in enumerate(ds_train):
          with tf.GradientTape() as tape:
            logits=self.model(img_batch,training=True)
            l=self.loss_function(lbl_batch,logits)
          grads=tape.gradient(l,self.model.trainable_variables)
          self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables)) #gradients are applied using the optimizer.
          self.train_loss(l)
          self.train_accuracy(lbl_batch, logits)
          grads=np.array(grads)
          epochGrad=grads+epochGrad #gradient vectors are appended at each step. The norm of the gradient are calculated at the end of the epoch.
          counter+=1
          if not step %100: #debugging print function
            acc=self.train_accuracy.result()
            print("Loss: {} Accuracy: {}".format(l,acc))
        self.accArray.append(acc) #accuracy value at the end of the epoch is appended to the accArray.
        self.lossArray.append(l)  #loss value at the end of the epoch is appended to the lossArray.
        epochGrad=epochGrad/counter #the accumulated gradient is divied by the number of step in each epoch. 
        normGradEpoch.append(calculateNorm(epochGrad)) #L2 norm of the gradient is calculated and appended to the normGradEpoch array.
        print("EPOCH: {} COMPLETED".format(i))
        print("###########")
      normGradEpoch=np.array(normGradEpoch)
      return normGradEpoch
