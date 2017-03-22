import tensorflow as tf


def make_hparam_string(learning_rate):
    return "lr_" + str(learning_rate)


# Add names to variables
def mnist(learning_rate, writer):

    # Define a simple convolutional layer
    def conv_layer(input, channels_in, channels_out, name="conv"):
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal([5, 5, channels_in, channels_out], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="B")
            conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
            act = tf.nn.relu(conv + b)
            tf.summary.histogram("weights", w)
            tf.summary.histogram("bias", b)
            tf.summary.histogram("activations", act)
            return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                  padding="SAME")

    # And a fully connected layer
    def fc_layer(input, channels_in, channels_out, name="fc"):
        with tf.name_scope(name):
            w = tf.Variable(tf.zeros([channels_in, channels_out]), name="W")
            b = tf.Variable(tf.zeros([channels_out]), name="B")
            act = tf.nn.relu(tf.matmul(input, w) + b)
            return act

    # Setup placeholders, and reshape the data
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 3)

    conv1 = conv_layer(x_image, 1, 32, "conv1")

    conv2 = conv_layer(conv1, 32, 64, "conv2")

    flattened = tf.reshape(conv2, [-1, 7 * 7 * 64])

    fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, "fc1")
    logits = fc_layer(fc1, 1024, 10, "fc2")

    # Compute cross entropy as our loss function
    with tf.name_scope("xent"):
        xent = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        tf.summary.scalar('cross_entropy', xent)

    # Use an AdamOptimizer to train the network
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

    # compute the accuracy
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # Initialize all the variables
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    merged_summary = tf.summary.merge_all()
    writer.add_graph(sess.graph)

    # Import MNIST data
    # Source: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network.ipynb
    # Import MNIST data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Train for 2000 steps
    for i in range(200):
        batch = mnist.train.next_batch(100)

        # Occasionally report accuracy
        if i % 20 == 0:
            s = sess.run(merged_summary, feed_dict={x: batch[0], y: batch[1]})
            writer.add_summary(s, i)
            [train_accuracy] = sess.run([accuracy], feed_dict={x: batch[0], y: batch[1]})
            print("step %d, training accuracy %g" % (i, train_accuracy))

        # Run the training step
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

    tf.reset_default_graph()


# Try a few learning rates
for learning_rate in [1e-3, 1e-4, 1e-5]:

    # construct a hyperparameter string for each one (example: "lr_1e-3")
    hparam_str = make_hparam_string(learning_rate)

    writer = tf.summary.FileWriter("./tmp/mnist_tutorial/" + hparam_str)

    # Actually run with the new settings
    mnist(learning_rate, writer)
