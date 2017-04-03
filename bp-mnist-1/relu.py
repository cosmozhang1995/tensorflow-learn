import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

class ReluMnistNet:

    def __init__(self, optimizer=None):
        self.varlist = []
        self.optimizer = optimizer or tf.train.GradientDescentOptimizer(0.01)

        # fetch dataset
        
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        
        # prepare environment
        layers = [ 100 ]
        input_layer = 784
        output_layer = 10
        
        self.x = tf.placeholder(tf.float32, [None, input_layer])
        last_layer = input_layer
        y = self.x
        for layer in layers:
            b = tf.Variable(tf.zeros([layer]))
            self.varlist.append(b)
            W = tf.Variable(tf.random_normal([last_layer,layer], stddev=0.01))
            self.varlist.append(W)
            y = tf.nn.relu( tf.matmul(y,W) ) + b
            last_layer = layer
        b = tf.Variable(tf.zeros([output_layer]))
        self.varlist.append(b)
        W = tf.Variable(tf.random_normal([last_layer,output_layer], stddev=0.01))
        self.varlist.append(W)
        self.y = tf.matmul(y,W) + b
        
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        
        self.cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_) )

    def prepare(self):
        init = tf.initialize_variables(self.varlist)
        
        self.sess = tf.Session()
        self.sess.run(init)
        
    def run(self, batch_size=100, stop=0.001, print_epoch=False):
        mnist = self.mnist
        data_size = mnist.train.images.shape[0]

        last_accuracy = 0

        accuracy_history = []

        train_step = self.optimizer.minimize(self.cross_entropy)
        for i in range(10000):
            for j in range(data_size/batch_size):
                # random batch
                batch_idx = np.arange(data_size)
                np.random.shuffle(batch_idx)
                batch_idx = batch_idx[0:batch_size]
                batch_xs = mnist.train.images[batch_idx]
                batch_ys = mnist.train.labels[batch_idx]
                # ordered batch
                # start = j * batch_size
                # end = (j+1) * batch_size
                # batch_xs, batch_ys = mnist.train.images[start:end], mnist.train.labels[start:end]
                self.sess.run(train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})
        
            # test the accuracy 
            correct_prediction = tf.equal( tf.argmax(self.y,1), tf.argmax(self.y_,1) )
            accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float32) )
            accuracy = self.sess.run(accuracy, feed_dict = {self.x: mnist.test.images, self.y_: mnist.test.labels})
            accuracy_history.append(accuracy)
            if print_epoch:
                print i, accuracy
            if last_accuracy != 0 and abs(last_accuracy-accuracy) < stop:
                break
            last_accuracy = accuracy

        return accuracy_history

    def close(self):
        if not (self.sess is None):
            self.sess.close()
            self.sess = None

if __name__ == '__main__':
    learner = ReluMnistNet()
    learner.optimizer = tf.train.GradientDescentOptimizer(0.01)
    for i in range(10):
        learner.prepare()
        learner.run(stop=0.01, print_epoch=True)
        learner.close()

