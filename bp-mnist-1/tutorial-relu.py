import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# fetch dataset

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# prepare environment

layers = [ 100 ]
input_layer = 784
output_layer = 10

x = tf.placeholder(tf.float32, [None, input_layer])
last_layer = input_layer
y = x
for layer in layers:
	b = tf.Variable(tf.zeros([layer]))
	W = tf.Variable(tf.random_normal([last_layer,layer], stddev=0.01))
	y = tf.nn.relu( tf.matmul(y,W) ) + b
	last_layer = layer
b = tf.Variable(tf.zeros([output_layer]))
W = tf.Variable(tf.random_normal([last_layer,output_layer], stddev=0.01))
y = tf.matmul(y,W) + b

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean( -tf.reduce_sum( y_*tf.log( tf.nn.softmax(y)), 1 ) )

# cross_entropy function from tensorflow
#cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=y, labels    =y_) )

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(100):
	# train each 128 batches
	for start,end in zip( range(0, 55000, 128), range(128, 55000, 128) ):
		batch_xs, batch_ys = mnist.train.images[start:end], mnist.train.labels[start:end]
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	# test the accuracy 
	correct_prediction = tf.equal( tf.argmax(y,1), tf.argmax(y_,1) )
	accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float32) )
	print i, sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels})

