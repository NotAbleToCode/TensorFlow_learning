import tensorflow as tf
g1 = tf.Graph()
with g1.as_default():
    a = tf.Variable([[1,2]])
    b = tf.Variable([[2],[3]])
    c = tf.matmul(a,b)
    c = a * c
    init_op = tf.global_variables_initializer()
with tf.Session(graph = g1) as sess:
    sess.run(init_op)
    print(sess.run(c))
