import tensorflow as tf
import edward.models as md
import edward.inferences as inf
import util.pnet_tokenize as tok


dists = {
    'NORMAL': md.Normal
}

acts = {
    'RELU': tf.nn.relu,
    'SOFTMAX': tf.nn.softmax
}

infs = {
    'KLqp': inf.KLqp,
    'KLpq': inf.KLpq
}

outtypes = {
    'CATEGORICAL': md.Categorical,
    'NORMAL': lambda x: md.Normal(x, 0.1),

}


class BNN:

    def __init__(self, spec_file):
        self.graph = tf.Graph()
        self.spec = tok.tokenize_net(spec_file)
        self.ff = 0
        self.conv = 0
        self.rnn = 0
        self.__gen_net__()

    def inference(self, x_train, y_train, n_iter):
        self.inftype(self.weights, data={
            self.x: x_train,
            self.y: y_train
        }).run(n_iter=n_iter)

    def sample(self):
        return self.y.sample()

    def __gen_net__(self):
        self.weights = {}
        in_spec = self.spec[0]
        shape = in_spec["params"]
        self.x = tf.placeholder(tf.float32, [None, *shape])
        x = self.x

        for sp in self.spec[1:-1]:
            if sp["ltype"] == "FF":
                x = self.add_ff(sp, x)

        self.y = outtypes[self.spec[-1]["dists"][0]](x)
        self.inftype = infs[self.spec[-1]["lspec"]]

    def add_conv(self):
        with self.graph.as_default():
            pass

    def add_ff(self, spec, x):
        self.ff += 1
        with tf.variable_scope("ff" + str(self.ff)):
            shape = spec["params"]
            [_, outs] = shape
            W = dists[spec["dists"][0]](
                loc=tf.zeros(shape, tf.float32),
                scale=tf.ones(shape, tf.float32)
            )
            self.weights[W] = dists[spec["dists"][0]](
                loc=tf.get_variable("qW/loc", shape, dtype=tf.float32),
                scale=tf.nn.relu(tf.get_variable("qW/scale", shape, dtype=tf.float32))
            )
            b = dists[spec["dists"][1]](
                loc=tf.zeros(outs, tf.float32),
                scale=tf.zeros(outs, tf.float32)
            )
            self.weights[b] = dists[spec["dists"][0]](
                loc=tf.get_variable("qb/loc", outs, tf.float32),
                scale=tf.nn.relu(tf.get_variable("qb/scale", outs, tf.float32))
            )

            return acts[spec["act"]](tf.matmul(x, W) + b)

    def add_rnn(self):
        pass

    def add_norm(self):
        pass

    def add_pool(self):
        pass
