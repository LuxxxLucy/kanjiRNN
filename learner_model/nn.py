"""
Various tensorflow utilities
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope

def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
    # eq # 24 and 25 of http://arxiv.org/abs/1308.0850
    norm1 = tf.sub(x1, mu1)
    norm2 = tf.sub(x2, mu2)
    s1s2 = tf.mul(s1, s2)
    z = tf.square(tf.div(norm1, s1))+tf.square(tf.div(norm2, s2))-2*tf.div(tf.mul(rho, tf.mul(norm1, norm2)), s1s2)
    negRho = 1-tf.square(rho)
    result = tf.exp(tf.div(-z,2*negRho))
    denom = 2*np.pi*tf.mul(s1s2, tf.sqrt(negRho))
    result = tf.div(result, denom)
    return result
