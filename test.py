import tensorflow as tf
import tensorjet as tj
import numpy as np
np.random.seed(42)

X_nk = tf.placeholder(shape=(3, 3), dtype='float32')
y_nk = tf.placeholder(shape=(3, 1), dtype='float32')
w_k = tf.placeholder(shape=(3, 1), dtype='float32')
b = tf.placeholder(shape=(), dtype='float32')

optim_config = tf.OptimizerOptions(do_common_subexpression_elimination=True, 
								   do_constant_folding=True,
								   do_function_inlining=True) 

graph_config = tf.GraphOptions(optimizer_options=optim_config)
config = tf.ConfigProto(graph_options=graph_config)

X_i, y_i, w_i, b_i = np.random.rand(3, 3).astype('float32'), \
					 np.random.rand(3, 1).astype('float32'), \
					 np.random.rand(3, 1).astype('float32'), \
					 np.array(0.23, dtype='float32')

t = b * b * b * b

with tf.Session(config=config) as sess:
	cb = tj.ClassBuilder(out=[t], g=t.graph)
	cb.build()