import tensorflow as tf
import numpy as np
g1 = tf.Graph()
with g1.as_default():
    with tf.variable_scope('zyy'):
        v=tf.get_variable("v",[1],initializer=tf.constant_initializer([[0.1]]))
        print(type(v))
        c = tf.Variable([1,2])
        print(type(c))
with tf.Session(graph = g1) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(v))
