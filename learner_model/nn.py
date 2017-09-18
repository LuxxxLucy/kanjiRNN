"""
Various tensorflow utilities
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope

def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
    # eq # 24 and 25 of http://arxiv.org/abs/1308.0850
    norm1 = tf.subtract(x1, mu1)
    norm2 = tf.subtract(x2, mu2)
    s1s2 = tf.multiply(s1, s2)
    z = tf.square(tf.div(norm1, s1))+tf.square(tf.div(norm2, s2))-2*tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2)
    negRho = 1-tf.square(rho)
    result = tf.exp(tf.div(-z,2*negRho))
    denom = 2*np.pi*tf.multiply(s1s2, tf.sqrt(negRho))
    result = tf.div(result, denom)
    return result

def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, x1_data, x2_data, pen_data,stroke_importance_factor):
    result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
    # implementing eq # 26 of http://arxiv.org/abs/1308.0850
    epsilon = 1e-20
    result1 = tf.multiply(result0, z_pi)
    result1 = tf.reduce_sum(result1, 2, keep_dims=True)
    result1 = -tf.log(tf.maximum(result1, 1e-20)) # at the beginning, some errors are exactly zero.
    result_shape = tf.reduce_mean(result1)

    result2 = tf.nn.softmax_cross_entropy_with_logits(logits=z_pen, labels=pen_data)
    pen_data_weighting = pen_data[:,:, 2]+np.sqrt(stroke_importance_factor)*pen_data[:, :,  0]+stroke_importance_factor*pen_data[:, :, 1]
    result2 = tf.multiply(result2, pen_data_weighting)
    result_pen = tf.reduce_mean(result2)

    result = result_shape + result_pen
    return result, result_shape, result_pen

# below is where we need to do MDN splitting of distribution params
def get_mixture_coef(output):
    # returns the tf slices containing mdn dist params
    # ie, eq 18 -> 23 of http://arxiv.org/abs/1308.0850
    z = output
    z_pen = z[:, :, 0:3] # end of stroke, end of character/content, continue w/ stroke
    z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split( z[:,:,3:], 6, axis=2)

    # process output z's into MDN paramters

    # softmax all the pi's:
    max_pi = tf.reduce_max(z_pi, 2, keep_dims=True)
    z_pi = z_pi - max_pi
    z_pi = tf.exp(z_pi)
    normalize_pi = 1/(tf.reduce_sum(z_pi, 2, keep_dims=True))
    z_pi = tf.multiply(normalize_pi, z_pi)

    # exponentiate the sigmas and also make corr between -1 and 1.
    z_sigma1 = tf.exp(z_sigma1)
    z_sigma2 = tf.exp(z_sigma2)
    z_corr = tf.tanh(z_corr)

    return [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen]
