import util.bnn as bnn
import tensorflow as tf
import numpy as np

net = bnn.BNN("basic_bnn.txt")

x_train = np.expand_dims(np.linspace(0, np.pi, 20), 1)
y_train = np.sin(x_train)

net.inference(x_train, y_train, 1000)


sess = tf.Session()
print(sess.run(net.sample(), feed_dict={net.x: x_train}))
