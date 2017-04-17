import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# fetch dataset

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False)
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=True)

def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def bias_variable(shape, val=0.1):
    initial = tf.constant(val, shape=shape)
    return tf.Variable(initial)

# train

class MnistCNN:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        # self.x = tf.placeholder(tf.float32, [None, 784])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.yPredition = None
        self.variables = []
        self.layers = []
        self.init = None
        self.cross_entropy = None
        self.train = None
        self.accuracy = None
        self.sess = None

    # kernels: [ ( conv_width_or_height, channels, max_pool_width_or_height ) ]
    def construct(self, kernels=[(5,5,2), (5,25,2), (3,125,2)], fulls=[100]):
        y = self.x
        outChan = None
        outWidth = None
        outHeight = None
        self.layers = [y]
        self.variables = []

        # convlutional layers
        for kernel in kernels:
            kSize = kernel[0]
            kChan = kernel[1]
            pSize = kernel[2]
            yShape = y.shape.as_list()
            convKernel = weight_variable([kSize, kSize, yShape[3], kChan])
            y = tf.nn.conv2d( input = y, filter = convKernel, strides = [1,1,1,1], padding = "VALID" )
            self.layers.append(y)
            self.variables.append(convKernel)
            y = tf.nn.max_pool( value = y, ksize = [1,pSize,pSize,1], strides = [1,pSize,pSize,1], padding = "VALID" )
            self.layers.append(y)

        # here y's shape should be [batch, 1, 1, last_channel]
        yShape = y.shape.as_list()
        # reshape y to vector
        y = tf.reshape(y, [-1,yShape[3]])

        # full-connected layers
        lastNeuronNum = y.shape.as_list()[1]
        for neuronNum in fulls:
            W = weight_variable([lastNeuronNum,neuronNum])
            b = bias_variable([neuronNum])
            y = tf.matmul(y,W) + b
            y = tf.nn.relu(y)
            self.layers.append(y)
            self.variables.append(W)
            self.variables.append(b)
            lastNeuronNum = neuronNum

        # output layer (softmax)
        neuronNum = 10
        W = weight_variable([lastNeuronNum,neuronNum])
        b = bias_variable([neuronNum])
        y = tf.matmul(y,W) + b
        # y = tf.nn.softmax(y)
        self.layers.append(y)
        self.variables.append(W)
        self.variables.append(b)
        self.yPredition = y

    def evaluate(self):
        # self.cross_entropy = tf.reduce_mean( -tf.reduce_sum( self.y*tf.log(self.yPredition) , 1 ) )
        self.cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.yPredition) )
        self.variables.append(self.cross_entropy)
        self.train = tf.train.GradientDescentOptimizer(0.01).minimize(self.cross_entropy)
        correct_prediction = tf.equal( tf.argmax(self.yPredition,1), tf.argmax(self.y,1) )
        self.accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float32) )

    def initialize(self):
        # self.init = tf.variables_initializer(self.variables)
        self.init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(self.init)

    def prepare(self):
        self.construct()
        self.evaluate()
        self.initialize()

    def train_step(self, batch_size=100, evaluate=False):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        self.sess.run(self.train, feed_dict = {self.x: batch_xs, self.y: batch_ys})
        # print self.sess.run(self.yPredition, feed_dict = {self.x: batch_xs, self.y: batch_ys})
        if evaluate:
            return self.sess.run(self.accuracy, feed_dict = {self.x: batch_xs, self.y: batch_ys})

    def test_step(self):
        return self.sess.run(self.accuracy, feed_dict = {self.x: mnist.test.images, self.y: mnist.test.labels})

if __name__ == '__main__':
    batch_size = 100
    batch_num = mnist.train.images.shape[0] / batch_size
    net = MnistCNN()
    net.prepare()
    for epoch in range(100):
        a = net.test_step()
        print "Epoch %d: accuracy = %f" % (epoch, a)
        for i in range(batch_num):
            net.train_step()

