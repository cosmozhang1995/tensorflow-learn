import tensorflow as tf
import numpy as np
from relu import ReluMnistNet

learning_rates = [0.005, 0.01, 0.03, 0.05]

fp = open("result.txt", "w")

for lr in learning_rates:
    learner = ReluMnistNet()
    learner.optimizer = tf.train.GradientDescentOptimizer(lr)
    learner.prepare()
    result, time = learner.run()
    learner.close()
    final_result = result[len(result)-1]
    result_str = "SGD: \n\tLearning rate: %f\n\tTime used: %ds\n\tFinal accuracy: %f (train) ;  %f (test)" % (lr, time, final_result[0], final_result[1])
    fp.write(result_str + "\n")
    print result_str

alphas = [0.5, 0.9, 0.99]
alpha = 0.9
for lr in learning_rates:
    learner = ReluMnistNet()
    learner.optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=alpha)
    learner.prepare()
    result, time = learner.run()
    learner.close()
    final_result = result[len(result)-1]
    result_str = "SGD with momentum: \n\tLearning rate: %f\n\tMomentum: %f\n\tTime used: %ds\n\tFinal accuracy: %f (train) ;  %f (test)" % (lr, alpha, time, final_result[0], final_result[1])
    fp.write(result_str + "\n")
    print result_str
lr = 0.01
for alpha in alphas:
    learner = ReluMnistNet()
    learner.optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=alpha)
    learner.prepare()
    result, time = learner.run()
    learner.close()
    final_result = result[len(result)-1]
    result_str = "SGD with momentum: \n\tLearning rate: %f\n\tMomentum: %f\n\tTime used: %ds\n\tFinal accuracy: %f (train) ;  %f (test)" % (lr, alpha, time, final_result[0], final_result[1])
    fp.write(result_str + "\n")
    print result_str

alphas = [0.5, 0.9, 0.99]
alpha = 0.9
for lr in learning_rates:
    learner = ReluMnistNet()
    learner.optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=alpha, use_nesterov=True)
    learner.prepare()
    result, time = learner.run()
    learner.close()
    final_result = result[len(result)-1]
    result_str = "SGD with Nesterov-momentum: \n\tLearning rate: %f\n\tMomentum: %f\n\tTime used: %ds\n\tFinal accuracy: %f (train) ;  %f (test)" % (lr, alpha, time, final_result[0], final_result[1])
    fp.write(result_str + "\n")
    print result_str
lr = 0.01
for alpha in alphas:
    learner = ReluMnistNet()
    learner.optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=alpha, use_nesterov=True)
    learner.prepare()
    result, time = learner.run()
    learner.close()
    final_result = result[len(result)-1]
    result_str = "SGD with Nesterov-momentum: \n\tLearning rate: %f\n\tMomentum: %f\n\tTime used: %ds\n\tFinal accuracy: %f (train) ;  %f (test)" % (lr, alpha, time, final_result[0], final_result[1])
    fp.write(result_str + "\n")
    print result_str

alphas = [0.1, 0.3, 0.5]
alpha = 0.1
for lr in learning_rates:
    learner = ReluMnistNet()
    learner.optimizer = tf.train.AdagradOptimizer(lr, initial_accumulator_value=alpha)
    learner.prepare()
    result, time = learner.run()
    learner.close()
    final_result = result[len(result)-1]
    result_str = "AdaGrad: \n\tLearning rate: %f\n\tInitial accumulation: %f\n\tTime used: %ds\n\tFinal accuracy: %f (train) ;  %f (test)" % (lr, alpha, time, final_result[0], final_result[1])
    fp.write(result_str + "\n")
    print result_str
lr = 0.01
for alpha in alphas:
    learner = ReluMnistNet()
    learner.optimizer = tf.train.AdagradOptimizer(lr, initial_accumulator_value=alpha)
    learner.prepare()
    result, time = learner.run()
    learner.close()
    final_result = result[len(result)-1]
    result_str = "AdaGrad: \n\tLearning rate: %f\n\tInitial accumulation: %f\n\tTime used: %ds\n\tFinal accuracy: %f (train) ;  %f (test)" % (lr, alpha, time, final_result[0], final_result[1])
    fp.write(result_str + "\n")
    print result_str

alphas = [0.5, 0.9, 0.99]
alpha = 0.9
for lr in learning_rates:
    learner = ReluMnistNet()
    learner.optimizer = tf.train.RMSPropOptimizer(lr, decay=alpha)
    learner.prepare()
    result, time = learner.run()
    learner.close()
    final_result = result[len(result)-1]
    result_str = "RMSProp: \n\tLearning rate: %f\n\tInitial accumulation: %f\n\tTime used: %ds\n\tFinal accuracy: %f (train) ;  %f (test)" % (lr, alpha, time, final_result[0], final_result[1])
    fp.write(result_str + "\n")
    print result_str
lr = 0.01
for alpha in alphas:
    learner = ReluMnistNet()
    learner.optimizer = tf.train.RMSPropOptimizer(lr, decay=alpha)
    learner.prepare()
    result, time = learner.run()
    learner.close()
    final_result = result[len(result)-1]
    result_str = "RMSProp: \n\tLearning rate: %f\n\tInitial accumulation: %f\n\tTime used: %ds\n\tFinal accuracy: %f (train) ;  %f (test)" % (lr, alpha, time, final_result[0], final_result[1])
    fp.write(result_str + "\n")
    print result_str

beta1s = [0.5, 0.9, 0.99]
beta2s = [0.9, 0.99, 0.999]
beta1 = 0.9
beta2 = 0.999
for lr in learning_rates:
    learner = ReluMnistNet()
    learner.optimizer = tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2)
    learner.prepare()
    result, time = learner.run()
    learner.close()
    final_result = result[len(result)-1]
    result_str = "Adam: \n\tLearning rate: %f\n\tBeta1: %f, Beta2: %f\n\tTime used: %ds\n\tFinal accuracy: %f (train) ;  %f (test)" % (lr, beta1, beta2, time, final_result[0], final_result[1])
    fp.write(result_str + "\n")
    print result_str
lr = 0.01
beta1 = 0.9
for lr in learning_rates:
    learner = ReluMnistNet()
    learner.optimizer = tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2)
    learner.prepare()
    result, time = learner.run()
    learner.close()
    final_result = result[len(result)-1]
    result_str = "Adam: \n\tLearning rate: %f\n\tBeta1: %f, Beta2: %f\n\tTime used: %ds\n\tFinal accuracy: %f (train) ;  %f (test)" % (lr, beta1, beta2, time, final_result[0], final_result[1])
    fp.write(result_str + "\n")
    print result_str
lr = 0.01
beta2 = 0.999
for alpha in alphas:
    learner = ReluMnistNet()
    learner.optimizer = tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2)
    learner.prepare()
    result, time = learner.run()
    learner.close()
    final_result = result[len(result)-1]
    result_str = "Adam: \n\tLearning rate: %f\n\tBeta1: %f, Beta2: %f\n\tTime used: %ds\n\tFinal accuracy: %f (train) ;  %f (test)" % (lr, beta1, beta2, time, final_result[0], final_result[1])
    fp.write(result_str + "\n")
    print result_str


