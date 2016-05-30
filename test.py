import tensorflow as tf
import tensorjet as tj
import numpy as np
np.random.seed(42)

b = tf.placeholder(shape=(), dtype='float32')

t = b * b * b * b

with tf.Session() as sess:
	cb = tj.ClassBuilder(out=[t], g=t.graph)
	cb.build()