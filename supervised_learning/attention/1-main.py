#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention

np.random.seed(0)
tf.compat.v1.set_random_seed(0)

attention = SelfAttention(256)
print(attention.W)
print(attention.U)
print(attention.V)
s_prev = tf.convert_to_tensor(
    np.random.uniform(size=(32, 256)), dtype='float32')
hidden_states = tf.convert_to_tensor(
    np.random.uniform(size=(32, 10, 256)), dtype='float32')
context, weights = attention(s_prev, hidden_states)
print(context)
print(weights)
