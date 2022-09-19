import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
from tensorflow.keras import Model

"""
Custom highway layer class for fully connected networks. Non-linear activation type, number of outputs,
transformgateBias values and weight initializer are passed as an argument. This custom layer is used in 
highway networks.
"""
class HighwayFCBlock(tf.keras.layers.Layer):
  def __init__(self, num_outputs,activationType,init,transformBias):
    super(HighwayFCBlock, self).__init__()
    self.num_outputs = num_outputs #number of neurons in the layer. 
    self.activations=[] #Intermediate activation values of the transform gate are appended to this array for later visualization in the heatmap.
    self.outputs=[] #Intermediate output of the layer are appended to this array for later visualization in the heatmap.
    self.activationType=activationType #non-linear activation type of the layer.
    self.init=init #Boolean for choice of weight initialization. If True is passed, Glorot Uniform. If no, He Normal. 
    self.transformBias=transformBias #value of the transform gate bias

  def build(self, input_shape):

    initializer1=tf.keras.initializers.GlorotUniform()
    initializer2 =tf.keras.initializers.Zeros()
    

    if self.init==True: #If init is True, Glorot Uniform are applied for Wh and Wt.
        self.kernel1 = self.add_weight("Wh",shape=[int(input_shape[-1]),self.num_outputs],trainable=True,initializer=tf.keras.initializers.HeNormal())
        self.kernel2 = self.add_weight("Wt",shape=[int(input_shape[-1]),self.num_outputs],trainable=True,initializer=tf.keras.initializers.HeNormal())
    else: #If init is False, He Normal is applied.
        self.kernel1 = self.add_weight("Wh",shape=[int(input_shape[-1]),self.num_outputs],trainable=True,initializer=initializer1)
        self.kernel2 = self.add_weight("Wt",shape=[int(input_shape[-1]),self.num_outputs],trainable=True,initializer=initializer1)
    #self.kernel3 = self.add_weight("Wc",initializer='he_normal',shape=[int(input_shape[-1]),self.num_outputs],trainable=True)


    self.bias1=self.add_weight("biasH",shape=[int(input_shape[-1])],trainable=True,initializer=initializer2) 
    self.bias2=self.add_weight("biasT",initializer=tf.keras.initializers.Constant(value=self.transformBias),shape=[int(input_shape[-1])],trainable=True) #Transform gate bias is initialized by the argument that user passed.
    #self.bias2=self.add_weight("biasT",shape=[int(input_shape[-1])],trainable=True)
    #self.bias3=self.add_weight("biasC",shape=[int(input_shape[-1])],trainable=True)

  #Returns the activations array for the later on visualization process.
  def getActivations(self):
    return self.activations
  #Returns the outputs array for the later on visualization process.
  def getOutput(self):
    return self.outputs

  #This function will be called during forward pass. Skip connection and the input transformation is done in this function.
  def call(self, input):
    #The non-linear activation function name is given as String argument. Input transformation is done in the if else statement. 
    if self.activationType=="relu":
        res1=tf.nn.relu(tf.matmul(input,self.kernel1)+self.bias1)
    elif self.activationType=="sigmoid":
        res1=tf.nn.sigmoid(tf.matmul(input,self.kernel1)+self.bias1)
    elif self.activationType=="tanh":
        res1=tf.nn.tanh(tf.matmul(input,self.kernel1)+self.bias1)
    else:
        res1=tf.nn.relu(tf.matmul(input,self.kernel1)+self.bias1)

    res2=tf.nn.sigmoid(tf.matmul(input,self.kernel2)+self.bias2) #Transform gate operation is done.
    output=tf.math.add(tf.math.multiply(res1,res2),tf.math.multiply(input,1-res2)) #Gate value and the input transformation is multiplied to produce the output value.
    self.activations.append(res2) #activation value of the transform gate appended to the activations array
    self.outputs.append(output) #output appended to the outputs array
    #res3=tf.nn.sigmoid(tf.matmul(input,self.kernel3)+self.bias3)

    return output