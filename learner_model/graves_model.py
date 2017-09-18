"""
The core model
"""

import settings
from learner_model.nn import *
from learner_model.model_session import ModelSession

import tensorflow as tf
import keras
from keras.layers import LSTM,Dense,Masking

import random

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
            x = tf.placeholder(tf.float32, shape=[None, args.seq_length, 5], name='x')
            y = tf.placeholder(tf.float32, shape=[None, args.seq_length, 5], name='y')
            x_lengths = tf.placeholder(tf.int32, shape=[None], name='x_lengths')

            drop_rate = tf.placeholder(tf.float32, name="drop_rate")
            learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        network = LSTM(64, return_sequences=True)(x)
        network = LSTM(64, return_sequences=True)(network)
        network = LSTM(64, return_sequences=True)(network)

        num_classes = args.num_mixture*6 + 3
        # Linear activation, using outputs computed above
        y_ = Dense(num_classes)(network)

        with tf.variable_scope("result"):
            [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen] = get_mixture_coef(y_)

            o_pi=tf.identity(o_pi,name="o_pi")
            o_mu1=tf.identity(o_pi,name="o_mu1")
            o_mu2=tf.identity(o_pi,name="o_mu2")
            o_sigma1=tf.identity(o_pi,name="o_sigma1")
            o_sigma2=tf.identity(o_pi,name="o_sigma2")
            o_corr=tf.identity(o_pi,name="o_corr")
            o_pen=tf.identity(o_pi,name="o_pen")

        with tf.variable_scope("train"):
            flat_target_data = tf.identity(y)
            [x1_data, x2_data, eos_data, eoc_data, cont_data] = tf.split(flat_target_data, 5, 2)
            pen_data = tf.concat([eos_data, eoc_data, cont_data],2)

            [lossfunc, loss_shape, loss_pen] = get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, x1_data, x2_data, pen_data, args.stroke_importance_factor)
            cost = lossfunc

            loss= tf.identity(cost,name='loss')

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), args.grad_clip)
            optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate, epsilon=0.001)
            train_op = optimizer.apply_gradients(zip(grads, tvars), name='train_step',global_step=iteration)

        with tf.variable_scope("test"):
            predictions=tf.cast(y_,tf.int32)

    def sample(self,args,initial_data):
        num=args.sample_length
        temp_mixture=args.temperature
        temp_pen=args.temperature
        stop_if_eoc =args.stop_if_eoc

        def get_pi_idx(x, pdf):
            N = pdf.size
            accumulate = 0
            for i in range(0, N):
                accumulate += pdf[i]
                if (accumulate >= x):
                    return i
            print ('error with sampling ensemble')
            return -1

        def sample_gaussian_2d(mu1, mu2, s1, s2, rho):
            mean = [mu1, mu2]
            cov = [[s1*s1, rho*s1*s2], [rho*s1*s2, s2*s2]]
            x = np.random.multivariate_normal(mean, cov, 1)
            return x[0][0], x[0][1]

        prev_x = [it for it in initial_data]
        #prev_x[0, 0, 2] = 1 # initially, we want to see beginning of new stroke
        #prev_x[0, 0, 3] = 1 # initially, we want to see beginning of new character/content
        # prev_state = sess.run(self.cell.zero_state(self.args.batch_size, tf.float32))

        init_len=5
        strokes = np.zeros((num+init_len, 5), dtype=np.float32)
        strokes[:init_len,:]=initial_data[0]
        mixture_params = []

        for i in range(num):
            prev_x = np.asarray(prev_x,dtype=np.float32)
            feed = {self.x: prev_x}
            [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen] = self.session.run([self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2, self.corr, self.pen],feed)

            pi_pdf = o_pi[0][-1]
            if i > 1:
                pi_pdf = np.log(pi_pdf) / temp_mixture
                pi_pdf -= pi_pdf.max()
                pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()

            idx = get_pi_idx(random.random(), pi_pdf)

            pen_pdf = o_pen[0]
            if i > 1:
                pi_pdf /= temp_pen # softmax convert to prob
            pen_pdf -= pen_pdf.max()
            pen_pdf = np.exp(pen_pdf)
            pen_pdf /= pen_pdf.sum()

            pen_idx = get_pi_idx(random.random(), pen_pdf)
            eos = 0
            eoc = 0
            cont_state = 0

            if pen_idx == 0:
                eos = 1
            elif pen_idx == 1:
                eoc = 1
            else:
                cont_state = 1

            next_x1, next_x2 = sample_gaussian_2d(o_mu1[0,-1,idx], o_mu2[0, -1, idx], o_sigma1[0, -1 , idx], o_sigma2[0, -1, idx], o_corr[0, -1, idx])

            point = [next_x1, next_x2, eos, eoc, cont_state]
            strokes[i+init_len,:] = point

            params = [pi_pdf, o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_corr[0], pen_pdf]
            mixture_params.append(params)

            # early stopping condition
            if (stop_if_eoc and eoc == 1):
              strokes = strokes[0:i+1+init_len, :]
              break

            prev_x = [ [its for its in it] for it in prev_x ]
            prev_x[0]=prev_x[0][:-1]
            prev_x[0].append([next_x1, next_x2, eos, eoc, cont_state])

        strokes[:,0:2] *= args.scale_factor
        return strokes, mixture_params


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
        result= self.session.run(self.loss, feed_dict={self.x: x, self.y: y, self.drop_rate: 0.0})
        return result

    def test(self, x, y):
        x, y = self.preprocess_for_train(x, y)
        result= self.session.run(self.loss, feed_dict={self.x: x, self.y: y, self.drop_rate: 0.0, self.learning_rate: learning_rate})


        return result

    def sample_test(self):
        return self.session.run([self.pred, self.iteration])



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

    @property
    def pi(self):
        return self._tensor("result/o_pi:0")

    @property
    def mu1(self):
        return self._tensor("result/o_mu1:0")

    @property
    def mu2(self):
        return self._tensor("result/o_mu2:0")

    @property
    def sigma1(self):
        return self._tensor("result/o_sigma1:0")

    @property
    def sigma2(self):
        return self._tensor("result/o_sigma2:0")

    @property
    def corr(self):
        return self._tensor("result/o_corr:0")

    @property
    def pen(self):
        return self._tensor("result/o_pen:0")


    def _tensor(self, name):
        return self.session.graph.get_tensor_by_name(name)
