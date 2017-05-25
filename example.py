import argparse
import sys
import time
import tensorflow as tf

FLAGS = None

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
  learning_rate = 0.0005
  batch_size = 100
  training_epochs=20
  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      global_step = tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False);
      # Build model...
      x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
      # target 10 output classes
      y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")
      tf.set_random_seed(1);
      W1 = tf.Variable(tf.random_normal([784, 100]))
      W2 = tf.Variable(tf.random_normal([100, 10]))
      b1 = tf.Variable(tf.zeros([100]))
      b2 = tf.Variable(tf.zeros([10]))
      z2 = tf.add(tf.matmul(x,W1),b1)
      a2 = tf.nn.sigmoid(z2)
      z3 = tf.add(tf.matmul(a2,W2),b2)
      y  = tf.nn.softmax(z3)
      cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
      grad_op=tf.train.GradientDescentOptimizer(learning_rate);
      train_op = grad_op.minimize(cross_entropy, global_step=global_step)
			
      correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

      init_op=tf.global_variables_initializer();
      
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                               global_step=global_step,
			       init_op=init_op)
    begin_time=time.time();
    frequency=100;
    with sv.prepare_or_wait_for_session(server.target) as sess:
      start_time=time.time();
      for epoch in range(training_epochs):
        batch_count=int(mnist.train.num_examples/batch_size);
        count=0;
        for i in range(batch_count):
          batch_x, batch_y = mnist.train.next_batch(batch_size);
          _,cost=sess.run([train_op,cross_entropy],feed_dict={x:batch_x,y_:batch_y});
          print(cost);
    sv.stop();
    print("done");

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="seltera46:2222",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="seltera46:2223,seltera46:2224,seltera46:2225",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
