import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
from tensorflow.keras import Model
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import linalg as LA
from matplotlib.pyplot import figure
from .utilsProject import *
from .highwayFC import *


"""
Class for fully connected highway networks. Input shape, depth of the network,number of neurons in each layer, non-linear activation function
type, method of weight initialiation, transform gate bias values and the number of outputs are passed as argument. 
"""
class highwayFC_Model:
    def __init__(self,input_shape,numberofHighway,NeuronWidth,activation,init,transformBias,initializer,numberOutput):
        self.numberofHighway = numberofHighway #number of highway layers in the network
        self.input_shape=input_shape 
        self.accArray=[] #accuracy array, accuracy values of each epoch is appended to this array if custom train function is used. 
        self.lossArray=[] #loss array, loss values of each epoch is appended to this array if custom train function is used. 
        self.model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=input_shape)])  #Flatten layer is done first for the succesive fully connected layer and the highway layers. 
        self.initializer=initializer #weight initializer for the first and the last dense layer.
        self.model.add(tf.keras.layers.Dense(NeuronWidth,activation=activation,kernel_initializer=self.initializer)) #Dense layer is added, for reducing the dimensionality of the input.
        self.NeuronWidth=NeuronWidth #number of neurons in the highway layer.
        self.activation=activation #type of activation function used in each layer.
        self.init=init #Boolean for the custom highway layer weight initialization. If yes, Glorot Uniform is applied. If no, He Normal will be applied.
        self.transformBias=transformBias #the value of the transform gate bias
        self.depth=numberofHighway+3 #overall depth of the network. There is a +3 becuase of flatten, first dense(reducing the dimensionality of the input) and the final dense layer(for classification) layers.
        for i in range(numberofHighway): #highway layers are appended in this for loop.
          self.model.add(HighwayFCBlock(NeuronWidth,activation,self.init,self.transformBias))
        self.model.add(tf.keras.layers.Dense(numberOutput,kernel_initializer=self.initializer)) #Final dense layer, for the classification.

    #This function is used for plotting the transform gate biases in the heatmap plot. Name of the dataset and the min-max values of the heatmap color scale values are passed as argument.
    def plotHeatMapBias(self,name,vmin,vmax):
      biasArray=np.zeros(shape=(self.numberofHighway,self.NeuronWidth)) #creating empty array for transform gate biases.
      for i in range(2,2+self.numberofHighway): #retrive all transform gate biases in for loop. The loop starts from index 2 becuase first two layers are not highway layers.
        biasArray[i-2,:]=self.model.layers[i].weights[3].numpy() #Third value of a highway layer is transform gate bias. Therefore, this value is appended to the biasArray.
      ax = sns.heatmap(biasArray,cmap="coolwarm",vmin=vmin,vmax=vmax) #heatmap plot
      ax.set_xticks([0,10,20,30,40,50])
      ax.set_xticklabels([0,10,20,30,40, 50])
      ax.set_yticks([0,10,20,30,40,50])
      ax.set_yticklabels([0,10,20,30,40, 50])
      plt.xlabel("Block")
      plt.ylabel("\n"+name+"\n""\n""Depth")
      plt.show()

    #Initializing the loss function and the metrics of the performance of the network for training and the test data.
    #This function is used before custom train function. 
    def init_loss(self,):
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

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

    def test_step(self,ds_test):
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()
        predictions = self.model.predict(ds_test)
        true_categories = tf.concat([y for x, y in ds_test], axis=0)
        t_loss = self.loss_function(true_categories, predictions)
        self.test_loss(t_loss)
        self.test_accuracy(true_categories,predictions)

    #Initializing the optimizer for the network. Learning rate and the momenntum values are passed as argument.
    #This function is used before custom train function.  
    def init_optimizer(self,lr,momentum):
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
        
    #Function for compiling the model. This function is usef before TF fit function. Learning rate, momentum are passed as an argument.        
    def compileModel(self,learningRate,momentum):
        self.model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learningRate, momentum=momentum),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],)

    def train(self,numberofEpochs,ds_train):
      normGradEpoch=[] #creating empty list for the norm of the gradients.
      self.train_loss.reset_states()
      self.train_accuracy.reset_states()
      for i in range(numberofEpochs):
        counter=1
        epochGrad=np.zeros(((self.numberofHighway*4)+4,)) #creating empty array for the gradients at each step.
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