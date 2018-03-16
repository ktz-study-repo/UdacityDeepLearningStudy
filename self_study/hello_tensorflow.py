import tensorflow as tf

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output = sess.run(tf.constant('Hello Word!'))
    print(output)
    x = tf.placeholder(tf.string)

    output = sess.run(x, feed_dict={x: 'Hello World!'})
    print(output)

    number = sess.run(tf.cast(tf.constant(1.5), tf.int32))

    print(number)

    n_features = 120
    n_labels = 5

    weight = tf.Variable(tf.truncated_normal((n_features, n_labels)))
    zeros = tf.zeros(n_labels)
    print(sess.run(zeros))

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

# TODO: Print cross entropy from session
cross_entropy = -tf.reduce_sum(tf.multiply(one_hot, tf.log(softmax)))
cross = tf.multiply(one_hot, softmax)

with tf.Session() as sess:
    print(sess.run(cross_entropy, feed_dict={softmax: softmax_data, one_hot: one_hot_data}))
    print(sess.run(cross, feed_dict={softmax: softmax_data, one_hot: one_hot_data}))
