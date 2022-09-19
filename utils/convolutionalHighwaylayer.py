class HighwayConvolutionalBlock(tf.keras.layers.Layer):
  def __init__(self,depth,kernel_size):
    super(HighwayConvolutionalBlock, self).__init__()
    self.depth = depth
    self.kernel_size=kernel_size
    
  def build(self, input_shape):
    self.kernel1 = self.add_weight("Wh",initializer='he_normal',shape=[self.kernel_size[0],self.kernel_size[1],input_shape[3],self.depth],trainable=True)
    self.kernel2 = self.add_weight("Wt",initializer='he_normal',shape=[self.kernel_size[0],self.kernel_size[1],input_shape[3],self.depth],trainable=True)

    self.bias1=self.add_weight("biasH",shape=[self.depth],trainable=True)
    self.bias2=self.add_weight("biasT",initializer=tf.random_uniform_initializer(minval=-3, maxval=-1),shape=[self.depth],trainable=True)

  def call(self, input):
    res1=tf.nn.conv2d(input,self.kernel1,[1,1,1,1], padding='SAME')
    res1= tf.nn.bias_add(res1,self.bias1)
    res1=tf.nn.relu(res1)

    res2=tf.nn.conv2d(input,self.kernel2,[1,1,1,1], padding='SAME')
    res2= tf.nn.bias_add(res2,self.bias2)
    res2=tf.nn.sigmoid(res2)

    return tf.math.add(tf.math.multiply(res1,res2),tf.math.multiply(input,1-res2))