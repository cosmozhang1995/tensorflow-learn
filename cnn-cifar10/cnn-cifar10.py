import tensorflow as tf
import numpy as np
from cifar10 import dataset as cifar10dataset

# fetch dataset

def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def bias_variable(shape, val=0.0):
    initial = tf.constant(val, shape=shape)
    return tf.Variable(initial)

# train

class Cifar10CNN:
    def __init__(self):
        self.saver = None
        self.x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        # self.x = tf.placeholder(tf.float32, [None, 784])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.keep_prob = tf.placeholder(tf.float32)
        self.yPredition = None
        self.variables = []
        self.layers = []
        self.outs = {}
        self.init = None
        self.cross_entropy = None
        self.train = None
        self.accuracy = None
        self.sess = None

    # kernels: [ ( conv_width_or_height, channels, max_pool_width_or_height ) ]
    def construct(self, kernels=[(5,24,2), (5,72,2), (5,120,1)], fulls=[84]):
        y = self.x
        outChan = None
        outWidth = None
        outHeight = None
        self.outs = {}
        self.layers = [y]
        self.outs["y0"] = y
        self.variables = []

        cnt = 1

        # convlutional layers
        for kernel in kernels:
            kSize = kernel[0]
            kChan = kernel[1]
            pSize = kernel[2]
            yShape = y.shape.as_list()
            convKernel = weight_variable([kSize, kSize, yShape[3], kChan])
            convBias = bias_variable([kChan])
            y = tf.nn.conv2d( input = y, filter = convKernel, strides = [1,1,1,1], padding = "VALID" ) + convBias
            y = tf.nn.relu(y)
            self.layers.append(y)
            self.outs["y%dc"%cnt] = y
            self.variables.append(convKernel)
            self.outs["w%d"%cnt] = convKernel
            self.variables.append(convBias)
            self.outs["b%d"%cnt] = convBias
            y = tf.nn.max_pool( value = y, ksize = [1,pSize,pSize,1], strides = [1,pSize,pSize,1], padding = "VALID" )
            self.layers.append(y)
            self.outs["y%d"%cnt] = y
            cnt += 1

        # here y's shape should be [batch, 1, 1, last_channel]
        yShape = y.shape.as_list()
        # reshape y to vector
        y = tf.reshape(y, [-1,yShape[1]*yShape[2]*yShape[3]])

        # full-connected layers
        lastNeuronNum = y.shape.as_list()[1]
        for neuronNum in fulls:
            W = weight_variable([lastNeuronNum,neuronNum])
            b = bias_variable([neuronNum])
            y = tf.matmul(y,W) + b
            y = tf.nn.relu(y)
            self.layers.append(y)
            self.outs["y%d"%cnt] = y
            self.variables.append(W)
            self.outs["fc%dw"%cnt] = W
            self.variables.append(b)
            self.outs["fc%db"%cnt] = b
            lastNeuronNum = neuronNum
            cnt += 1

        # dropout layer
        y = tf.nn.dropout(y, self.keep_prob)
        self.layers.append(y)
        self.outs["y_dropout"] = y
        cnt += 1

        # output layer (softmax)
        neuronNum = 10
        W = weight_variable([lastNeuronNum,neuronNum])
        b = bias_variable([neuronNum])
        y = tf.matmul(y,W) + b
        # y = tf.nn.softmax(y)
        self.layers.append(y)
        self.outs["y_out"] = y
        self.variables.append(W)
        self.outs["fc_out_w"] = W
        self.variables.append(b)
        self.outs["fc_out_b"] = b
        self.yPredition = y

    def evaluate(self):
        # self.cross_entropy = tf.reduce_mean( -tf.reduce_sum( self.y*tf.log(self.yPredition) , 1 ) )
        self.cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.yPredition) )
        self.variables.append(self.cross_entropy)
        self.train = tf.train.GradientDescentOptimizer(0.01).minimize(self.cross_entropy)
        correct_prediction = tf.equal( tf.argmax(self.yPredition,1), tf.argmax(self.y,1) )
        self.accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float32) )

    def initialize(self, filename=None):
        self.saver = tf.train.Saver()
        # self.init = tf.variables_initializer(self.variables)
        self.init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(self.init)
        if filename:
            self.saver.restore(self.sess, filename)

    def prepare(self, filename=None):
        self.construct()
        self.evaluate()
        self.initialize(filename)

    def train_step(self, batch_size=100, evaluate=False, loss=False):
        # batch = cifar10dataset.train.next_batch(batch_size)
        # batch_xs = batch.data
        # batch_ys = batch.labels
        lst = np.array(range(50000))
        np.random.shuffle(lst)
        lst = lst[0:100]
        batch_xs = cifar10dataset.train.data[lst,:,:,:]
        batch_ys = cifar10dataset.train.labels[lst,:]
        # print batch_xs[:,16,16,1]
        self.sess.run(self.train, feed_dict = {self.x: batch_xs, self.y: batch_ys, self.keep_prob: 0.5})
        # print self.sess.run(self.yPredition, feed_dict = {self.x: batch_xs, self.y: batch_ys})
        # print "prediction", np.argmax( self.sess.run(self.yPredition, feed_dict = {self.x: batch_xs, self.y: batch_ys}) , axis=1)
        # print "label", np.argmax( batch_ys , axis = 1 )
        if evaluate:
            return self.sess.run([self.accuracy, self.cross_entropy], feed_dict = {self.x: batch_xs, self.y: batch_ys, self.keep_prob: 1})

    def test_step(self):
        return self.sess.run([self.accuracy, self.cross_entropy], feed_dict = {self.x: cifar10dataset.test.data, self.y: cifar10dataset.test.labels, self.keep_prob: 1})

    def outNames(self):
        return self.outs.keys()

    def readParam(self, paramName):
        val = self.sess.run(self.outs[paramName])
        return val

    def trainEpoches(self, epoches=1):
        batch_size = 100
        batch_num = cifar10dataset.train.data.shape[0] / batch_size
        for epoch in range(epoches):
            for i in range(batch_num):
                self.train_step()
        a = self.test_step()[0]

    def save(self, filename):
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.saver.save(self.sess, filename)

def prepareNet(filename = None):
    net = MnistCNN()
    net.prepare(filename)
    return net


if __name__ == '__main__':
    if len(sys.argv) == 1:
        tf.device("/gpu:1")
        batch_size = 100
        batch_num = cifar10dataset.train.data.shape[0] / batch_size
        net = MnistCNN()
        start_epoch = 0
        if os.path.exists(save_dir):
            net.prepare(save_path)
            f = open(save_epoch_path)
            start_epoch = int(f.read())
            f.close()
            print "State loaded from: %s" % save_path
        else:
            net.prepare()
        for epoch in range(start_epoch, 1000):
            test = net.test_step()
            accuracy = test[0]
            loss = test[1]
            print "Epoch %d: accuracy = %f , loss = %f" % (epoch, accuracy, loss)
            if (epoch % 10) == 0:
                net.save(save_path)
                f = open(save_epoch_path, "w")
                f.write("%d"%epoch)
                f.close()
                print "State saved to: %s" % save_path
            for i in range(batch_num):
                net.train_step()
    elif sys.argv[1] == "w":
        if os.path.exists(save_dir):
            n = prepareNet(save_path)
            print "Accuracy: %f" % n.test_step()[0]
            print "Params:", n.outNames()
            w1 = n.readParam('w1')
            w2 = n.readParam('w2')
            im = n.sess.run(n.outs['y1c'], feed_dict = {n.x: cifar10dataset.test.data[0:1,:,:,:]})
            import scipy.io as sio
            sio.savemat("/home/cosmo/downloads/w.mat", {'w1':w1, 'w2':w2, 'im':im})
        else:
            n = prepareNet()
            w0 = n.readParam('w1')
            print "Accuracy: %f" % n.trainEpoches(5)
            print "Params:", n.outNames()
            w1 = n.readParam('w1')
            w2 = n.readParam('w2')
            im = n.sess.run(n.outs['y1c'], feed_dict = {n.x: cifar10dataset.test.data[0,:,:,:]})
            import scipy.io as sio
            sio.savemat("/home/cosmo/downloads/w.mat", {'w0':w0, 'w1':w1, 'w2':w2, 'im':im})