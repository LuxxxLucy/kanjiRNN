"""
The core model
"""

import settings
from learner_model.nn import *
from learner_model.model_session import ModelSession

import tensorflow as tf
import keras
from keras.layers import LSTM,Dense,Masking

# print  function for debug
from pprint import pprint as pr

savePath = settings.MODEL_STORE_PATH

ITEM_DIM = 100

class LSTM_Model_Session(ModelSession):
    """
    An OOP style ModelSession for rnn model
    """

    @classmethod
    def create(cls, **kwargs):
        """
        Create a new model session.

        :param kwargs: optional graph parameters
        :type kwargs: dict
        :return: new model session
        :rtype: ModelSession
        """
        session = tf.Session()
        from keras import backend as K
        K.set_session(session)
        with session.graph.as_default():
            cls.create_graph(**kwargs)
        session.run(tf.global_variables_initializer())
        return cls(session, tf.train.Saver(), kwargs['args'])

    @staticmethod
    def create_graph(args=None, infer=False):
        """
        The fully connected model for predicting next item that would be consumed by the user

        :param class_num: The class number, which means the number of different films' types
        :param item_dim: The item's dimension after embedding
        :param val_portion: Determine the number of validation data
        :param save_path: The path to save the model
        :param n_epoch: The number of epochs for training
        :param batch_size: The number of records in each batch
        :param learning_rate: The learning rate
        :param print_freq: The frequency to print the information
        :param global_step: The global step used for continuous training
        :param layer_num: The number of fully-connected NN layer
        :param top_unit_num: The number of units in the top layer of this NN
        :param dropout: The dropout rate of each layer

        """

        if infer:
            args.batch_size = 1
            args.seq_length = 1


        iteration = tf.Variable(initial_value=0, trainable=False, name="iteration")

        with tf.variable_scope("parameters"):
            x = tf.placeholder(tf.float32, shape=[args.batch_size, args.seq_length, 5], name='x')
            y = tf.placeholder(tf.float32, shape=[args.batch_size, args.seq_length, 5], name='y')
            x_lengths = tf.placeholder(tf.int32, shape=[args.batch_size], name='x_lengths')

            drop_rate = tf.placeholder(tf.float32, name="drop_rate")
            learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        network = LSTM(64, return_sequences=True)(x)
        network = LSTM(64, return_sequences=True)(network)
        network = LSTM(64, return_sequences=True)(network)

        num_classes = args.num_mixture*6 + 3
        # Linear activation, using outputs computed above
        y_ = Dense(num_classes)(network)


        with tf.variable_scope("train"):
            flat_target_data = tf.identity(y)
            [x1_data, x2_data, eos_data, eoc_data, cont_data] = tf.split(flat_target_data, 5, 2)
            pen_data = tf.concat([eos_data, eoc_data, cont_data],2)

            [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen] = get_mixture_coef(y_)

            [lossfunc, loss_shape, loss_pen] = get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, x1_data, x2_data, pen_data, args.stroke_importance_factor)
            cost = lossfunc

            loss= tf.identity(cost,name='loss')

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), args.grad_clip)
            optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate, epsilon=0.001)
            train_op = optimizer.apply_gradients(zip(grads, tvars), name='train_step',global_step=iteration)

        with tf.variable_scope("test"):
            predictions=tf.cast(y_,tf.int32)

    def __str__(self):
        return "LSTM Model (Graves et al.) (iteration %d)" % (
            self.session.run(self.iteration))

    def preprocess_for_train(self, X, y):
        return self.preprocess(X,y)

    def preprocess(self, X, y):
        # max_length = 30
        lengths = []
        for record in X:
            lengths.append(len(record))

        result_x = np.asarray(X, dtype=np.float32)
        result_y = np.asarray(y, dtype=np.float32)
        lengths = np.asarray(lengths, dtype=np.int32)
        return result_x, result_y, lengths

    def preprocess_for_test(self, X, y):
        return X, y

    def train(self, X_train_a, y_train_a, learning_rate, drop_rate=0.0):
        x, y, seq_len = self.preprocess_for_train(X_train_a, y_train_a)
        return self.session.run([self.train_step, self.iteration],
                                feed_dict={self.x: x,
                                           self.y: y,
                                           self.drop_rate: drop_rate,
                                           self.seq_len:seq_len,
                                           self.learning_rate: learning_rate})[1]

    def test_batch(self, x, y):
        x, y, seq_len = self.preprocess_for_train(x, y)
        result= self.session.run(self.loss, feed_dict={self.x: x, self.y: y, self.drop_rate: 0.0, self.learning_rate: learning_rate})
        return result

    def test(self, x, y):
        x, y = self.preprocess_for_train(x, y)
        result= self.session.run(self.loss, feed_dict={self.x: x, self.y: y, self.drop_rate: 0.0, self.learning_rate: learning_rate})
        return result


    @property
    def train_step(self):
        return self._tensor("train/train_step:0")

    @property
    def loss(self):
        return self._tensor("train/loss:0")

    @property
    def y_(self):
        return self._tensor("y_result:0")


    @property
    def iteration(self):
        return self._tensor("iteration:0")

    @property
    def x(self):
        return self._tensor("parameters/x:0")

    @property
    def y(self):
        return self._tensor("parameters/y:0")

    @property
    def seq_len(self):
        return self._tensor("parameters/x_lengths:0")

    @property
    def y_test(self):
        return self._tensor("parameters/y_test:0")

    @property
    def drop_rate(self):
        return self._tensor("parameters/drop_rate:0")

    @property
    def learning_rate(self):
        return self._tensor("parameters/learning_rate:0")

    def _tensor(self, name):
        return self.session.graph.get_tensor_by_name(name)
