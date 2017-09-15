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
    def create_graph(args, infer=False):
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
            x = tf.placeholder(tf.float64, shape=[args.batch_size, args.seq_length, 5], name='x')
            y = tf.placeholder(tf.float64, shape=[args.batch_size, args.seq_length, 5], name='y')
            X_lengths = tf.placeholder(tf.float64, shape=[args.batch_size, args.seq_length,], name='x')
            drop_rate = tf.placeholder(tf.float32, name="drop_rate")
            learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        cell = tf.nn.rnn_cell.LSTMCell(num_units=64, state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=1-drop_rate)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell] * 4, state_is_tuple=True)

        # network = tf.layers.dropout(inputs=network, rate=drop_rate)
        # y_output = tf.layers.dense(inputs=network, units=class_num)
        # y_ = tf.identity(y_output, name='y_result')

        outputs, last_states = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float64, sequence_length=x_lengths, inputs=x)


        # outputs, states  = tf.nn.bidirectional_dynamic_rnn( cell_fw=cell, cell_bw=cell, dtype=tf.float64, sequence_length=X_lengths, inputs=X)
        #
        # output_fw, output_bw = outputs
        # states_fw, states_bw = states


        with tf.variable_scope("train"):
            cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_, labels=y))
            train_op = tf.train.AdagradOptimizer(learning_rate,initial_accumulator_value=0.1,
                                              use_locking=False).minimize(cost, global_step=iteration, name="train_step")
            y_predict = tf.argmax(y_, 1)
            correct_prediction = tf.equal(y_predict, y)
            acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

        with tf.variable_scope("test"):

            in_top_10 = tf.nn.in_top_k(predictions=y_, targets=y_test[:,0], k=10, name=None)
            sps = tf.reduce_mean(tf.cast(in_top_10, tf.float32),name="sps")

            truth_length=tf.nn.relu(get_length(y_test),name='truth_length')
            predictions=tf.nn.top_k(y_, k=10, sorted=True)[1]
            predictions=tf.cast(predictions,tf.int64)

            hit_ones=tf.sets.set_intersection(predictions,y_test)
            hit_ones=tf.sparse_tensor_to_dense(hit_ones,name='hit_length')
            hit_length=get_length(hit_ones,false_value=0)
            recall =  tf.reduce_mean(tf.divide(hit_length, truth_length),name='recall')


    def __str__(self):
        return "LSTM Model (Graves et al.) (iteration %d)" % (
            self.session.run(self.iteration))

    def preprocess_for_train(self, X, y):

        X=X.tolist()
        y=y.tolist()

        target_x=[]
        target_y=[]
        for tmp_x,tmp_y in zip(X,y):
            for i in range(self.args.max_len):
                if tmp_x[i] != '0':
                    record = tmp_x[:i] + ['0'] * (self.args.max_len-i)
                    target_x.append(record)
                    target_y.append(tmp_x[i])
                else:
                    continue
            record = tmp_x
            target_x.append(record)
            target_y.append(tmp_y)

        return self.preprocess(target_x,target_y)

    def preprocess(self, X, y):

        max_length=20

        result_x = []
        for record in X:
            temp = [ int(it)-1 for it in record ]
            result_x.append(temp)

        result_y = []
        for y_train_a in y:
            result_y.append(int(y_train_a)-1)

        result_x = np.asarray(result_x, dtype=np.float32)
        result_y = np.asarray(result_y, dtype=np.int32)

        return result_x, result_y

    def preprocess_for_test(self, X, y):

        max_length=20

        result_x = []
        for record in X:
            temp = [ int(it)-1 for it in record ]
            result_x.append(temp)

        result_y = []
        for record in y:
            temp = [ int(it)-1 for it in record ]
            result_y.append(temp)

        result_x = np.asarray(result_x, dtype=np.float32)
        result_y = np.asarray(result_y, dtype=np.int32)

        return result_x, result_y

    def train(self, X_train_a, y_train_a, learning_rate, drop_rate=0.0):
        x, y = self.preprocess_for_train(X_train_a, y_train_a)
        return self.session.run([self.train_step, self.iteration],
                                feed_dict={self.x: x,
                                           self.y: y,
                                           self.drop_rate: drop_rate,
                                           self.learning_rate: learning_rate})[1]

    def test_batch(self, x, y):
        x, y = self.preprocess_for_train(x, y)
        result= self.session.run(self.accuracy, feed_dict={self.x: x, self.y: y, self.drop_rate: 0.0})
        return result

    def test(self, x, y):
        x, y = self.preprocess_for_test(x, y)
        result= self.session.run([self.sps,self.recall], feed_dict={self.x: x, self.y_test: y, self.drop_rate: 0.0})

        return result

    def test_top_n(self, x, y, n=5):
        y= [ it[0] for it in y]
        x, y = self.preprocess(x,y)
        # self.session.run(self.accuracy, feed_dict={self.x: x, self.y: y})
        feed_dict = {self.x: x, self.y: y, self.drop_rate: 0.0}
        # compute the top-k value evaluation
        temp_n = tf.nn.in_top_k(predictions=self.y_, targets=self.y, k=n, name=None)
        temp_n_ = self.session.run(temp_n, feed_dict=feed_dict)

        return sum([ 1 for it in temp_n_ if it == True] ) / len(temp_n_)



    @property
    def train_step(self):
        return self._tensor("train/train_step:0")

    @property
    def accuracy(self):
        return self._tensor("train/accuracy:0")

    @property
    def sps(self):
        return self._tensor("test/sps:0")

    @property
    def recall(self):
        return self._tensor("test/recall:0")

    @property
    def truth_length(self):
        return self._tensor("test/truth_length:0")

    @property
    def hit_length(self):
        return self._tensor("test/hit_length:0")

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
