import util.bnn as bnn
import tensorflow as tf
import numpy as np
import edward as ed

net = bnn.BNN("basic_bnn.txt")

x_train = np.expand_dims(np.linspace(0, np.pi, 20, dtype=np.float32), 1)
xmean = x_train.mean()
xstd = x_train.std()
x_train = (x_train - xmean) / xstd
y_train = np.sin(x_train)
ymean = y_train.mean()
ystd = y_train.std()
y_train = (y_train - ymean) / ystd

net.inference(x_train, y_train, 1000)

y_post = ed.copy(net.y, net.weights)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

output = sess.run(y_post, feed_dict={net.x: x_train})

print(ed.evaluate('mae', data={net.x: x_train, y_post: y_train}))
print(output)
print(y_train)
