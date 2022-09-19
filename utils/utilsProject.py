import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
from tensorflow.keras import Model
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import linalg as LA
from matplotlib.pyplot import figure
from .highwayFC import *


#Function for calculating the L2 norm of the gradient given gradient array.
def calculateNorm(gradients):
  gradientNormArray=[]
  for i in range(gradients.shape[0]):
    gradientNormArray.append(LA.norm(gradients[i]))
  return np.array(gradientNormArray)
#This function is used for plotting the norm of the gradients calculated at the end of each epoch. Layer indexes are passed as argument in indexArray. 
def plotMultipleGradientOverEpoch(normGrad,indexArray):
  plt.figure(figsize=(8,8))
  globalTemp=[]
  for j in range(len(indexArray)):
    temp=[]
    for i in range(normGrad.shape[0]):
      temp.append(normGrad[i][indexArray[j]])
    temp=np.array(temp)
    globalTemp.append(temp)
  globalTemp=np.array(globalTemp)
  for i in range(globalTemp.shape[0]):
    plt.plot([*range(1,normGrad.shape[0]+1,1)],globalTemp[i],label="Parameter index: "+str(indexArray[i]))
  plt.xlabel("number of epochs")
  plt.ylabel("gradient norm")
  plt.legend()

# Passed data set is the input of the model, whihch generates the output and calculates the mean of the transform gate output, and generates a heatmap plot
def meantransformGateOutputHeatMap(model,data,name,vmin,vmax):
  blockOutput=np.zeros((50,50))
  temp=model(data)
  for i in range(2,52):
    if name=="mnist":
      blockOutput[i-2]=np.sum(model.layers[i].getActivations()[-1],axis=0)/60000
    else:
      blockOutput[i-2]=np.sum(model.layers[i].getActivations()[-1],axis=0)/50000
  ax = sns.heatmap(blockOutput,cmap="coolwarm",vmin=vmin,vmax=vmax)
  ax.set_xticks([0,10,20,30,40,50])
  ax.set_xticklabels([0,10,20,30,40, 50])
  ax.set_yticks([0,10,20,30,40,50])
  ax.set_yticklabels([0,10,20,30,40, 50])
  plt.xlabel("Block")
  plt.ylabel("\n"+name+"\n""\n""Depth")
  plt.show()

# For a single sampled input calculate the transform gate output for each  highway layers, and generates heatmap plot
def transformGateOutputHeatMap(model,test,name):
  temp=model(test)
  blockOutput=np.zeros((50,50))
  for i in range(2,52):
    blockOutput[i-2]=model.layers[i].getActivations()[-1]
  ax = sns.heatmap(blockOutput,cmap="coolwarm",vmin=0,vmax=1)
  ax.set_xticks([0,10,20,30,40,50])
  ax.set_xticklabels([0,10,20,30,40, 50])
  ax.set_yticks([0,10,20,30,40,50])
  ax.set_yticklabels([0,10,20,30,40, 50])
  plt.xlabel("Block")
  plt.ylabel("\n" +name+"\n""\n""Depth")
  plt.show()

# For a single sampled input calculate the block output for each highway layers, and generates heatmap plot
def blockOutputHeatmap(model,test,name):
  activationArray=[]
  blockOutput=np.zeros((50,50))
  for i in range(2,52):
    base_model=model
    x=model.input
    x = base_model.layers[i].output
    model1=Model(inputs=base_model.input,outputs=x)
    activationArray.append(model1(test))
  for i in range(50):
    blockOutput[i]=activationArray[i]
  ax = sns.heatmap(blockOutput,cmap="coolwarm",vmin=-1,vmax=1)
  ax.set_xticks([0,10,20,30,40,50])
  ax.set_xticklabels([0,10,20,30,40, 50])
  ax.set_yticks([0,10,20,30,40,50])
  ax.set_yticklabels([0,10,20,30,40, 50])
  plt.xlabel("Block")
  plt.ylabel("\n" +name+ "\n""\n""Depth")
  plt.show()

# This function is used for removing a single layer from the given model. The index of the layer is passed as an argument. 
def removeMidlayer(model,no):
  base_model=model
  x=base_model.layers[0].input
  for i in range(len(base_model.layers)):
    if i!=no:
      x = base_model.layers[i](x)
  result_model = Model(inputs=base_model.layers[0].input, outputs=x)
  return result_model

# For each layer removes it from the model and Calculates the loss value and append it to the lossArray
def plotLesionedGraph(model,true_categories,data):
  loss_function=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  lossArray=[]
  for i in range(2,len(model.layers)-1):
    tempModel=removeMidlayer(model,i)
    preds=tempModel.predict(data,batch_size=512)
    t_loss = loss_function(true_categories, preds)
    lossArray.append(t_loss)
  lossArray=np.array(lossArray)
  fig, ax = plt.subplots(1, 1)
  ax.xaxis.set_ticks_position("top")
  ax.set_yscale("log")
  for i in range(len(lossArray)):
    plt.plot([*range(1,51)],lossArray,marker='o',color="mediumblue",markersize=3,linewidth=0.5)
  plt.xlabel("Lesioned Highway layer")
  plt.ylabel("Mean Cross Entropy Error")

"""
This function is used for calculating the mean and the variance of the activation at a single layer.
The layer index is passed as an argument for determing which layer is used. The data used for calculating this statistics is also passed as an argument.
"""
def activationMeanStdLayer(model,layerNo,ds_test):
  base_model=model
  x=base_model.input
  x = base_model.layers[layerNo].output
  model=Model(inputs=base_model.input,outputs=x)
  activationArray=[]
  for step,(img_batch,lbl_batch) in enumerate(ds_test):
    activationArray.append(model(img_batch,training=False)) 
  activationArray=np.array(activationArray)
  sum=np.zeros((activationArray[0].shape[1],))
  for i in range(activationArray.shape[0]-1):
    for j in range(activationArray[i].shape[0]):
      sum+=activationArray[i][j]
  sum=sum/((activationArray.shape[0]-1)*(activationArray[i].shape[0]))
  return np.mean(sum),np.std(sum)

# Using the `activateMeansStdLayer` returns the mean and variance of each layer in the network
def activationMeanStdLayerModel(model,ds_test,layerList):
  meanArray=[]
  stdArray=[]
  for i in layerList:
    m,s=activationMeanStdLayer(model,i,ds_test)
    meanArray.append(m)
    stdArray.append(s)
  meanArray=np.array(meanArray)
  stdArray=np.array(stdArray)
  return meanArray,stdArray

# Function for plotting the variance of the activation at each layer. Number of layers of the network is passed as an argument.
def plotstdActivation(stdArray,depth):
  plt.plot([*range(1,depth,1)],stdArray)
  plt.ylabel("Variance of the Activation")
  plt.xlabel("Layer no")

#Function for plotting mean activation at each layer. Number of layers of the network is passed as an argument.
def plotmeanActivation(meanArray,depth):
  plt.plot([*range(1,depth,1)],meanArray)
  plt.ylabel("Mean Activation")
  plt.xlabel("Layer no")