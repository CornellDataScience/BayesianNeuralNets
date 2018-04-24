import tensorflow as tf
import edward as ed
import util.pnet_tokenize as tok


class BasicBNN:

    def __init__(self, spec_file):
        self.sess = tf.Session()
        self.graph = tf.Graph()
        self.tokens = tok.tokenize_net(spec_file)
        self.__gen_net__()

    def __gen_net__(self):
        pass
