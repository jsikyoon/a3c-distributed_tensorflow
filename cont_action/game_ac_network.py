# -*- coding: utf-8 -*-
import tensorflow as tf
from game_state import STATE_SIZE
import numpy as np
import math

# Actor-Critic Network Base Class
# (Policy network and Value network)
class GameACNetwork(object):
  def __init__(self,
               action_size,
               thread_index, # -1 for global               
               device="/cpu:0"):
    self._action_size = action_size
    self._thread_index = thread_index
    self._device = device    

  def prepare_loss(self, entropy_beta):
    with tf.device(self._device):
      # taken action (input for policy)
      #self.a = tf.placeholder("float", [None, self._action_size])
    
      # temporary difference (R-V) (input for policy)
      self.td = tf.placeholder("float", [None])

      # avoid NaN with clipping when value in pi becomes zero
      log_prob = tf.log(tf.clip_by_value(self.prob, 1e-20, 1.0))
      
      # policy entropy
      #entropy = -tf.reduce_sum(self.pi * log_pi, reduction_indices=1)
      entropy = -0.5*(tf.log(self.sigma*2*math.pi)+1)
      
      # policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent, but we use gradient descent optimizer.)
      #policy_loss = - tf.reduce_sum( tf.reduce_sum( tf.multiply( log_prob, self.action ), reduction_indices=1 ) * self.td + tf.reduce_sum(entropy) * 0.0001 )
      policy_loss = tf.reduce_sum( tf.reduce_sum( log_prob ) * self.td + tf.reduce_sum(entropy) * 0.0001 )
      #policy_loss = - tf.reduce_sum( tf.reduce_sum( self.log_prob ) + tf.reduce_sum(entropy * 0.0001))

      # R (input for value)
      self.r = tf.placeholder("float", [None])
      
      # value loss (output)
      # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
      value_loss = 0.5*tf.nn.l2_loss(self.r - self.v)

      # gradienet of policy and value are summed up
      self.total_loss = policy_loss + value_loss

  def run_policy_and_value(self, sess, s_t):
    raise NotImplementedError()
    
  def run_policy(self, sess, s_t):
    raise NotImplementedError()

  def run_value(self, sess, s_t):
    raise NotImplementedError()    

  def get_vars(self):
    raise NotImplementedError()

  def sync_from(self, src_netowrk, name=None):
    src_vars = src_netowrk.get_vars()
    dst_vars = self.get_vars()

    sync_ops = []

    with tf.device(self._device):
      with tf.name_scope(name, "GameACNetwork", []) as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars):
          sync_op = tf.assign(dst_var, src_var)
          sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)

  # weight initialization based on muupan's code
  # https://github.com/muupan/async-rl/blob/master/a3c_ale.py
  def _fc_variable(self, weight_shape):
    input_channels  = weight_shape[0]
    output_channels = weight_shape[1]
    d = 1.0 / np.sqrt(input_channels)
    bias_shape = [output_channels]
    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
    bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
    return weight, bias

  def _conv_variable(self, weight_shape):
    w = weight_shape[0]
    h = weight_shape[1]
    input_channels  = weight_shape[2]
    output_channels = weight_shape[3]
    d = 1.0 / np.sqrt(input_channels * w * h)
    bias_shape = [output_channels]
    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
    bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
    return weight, bias

  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

# Actor-Critic FF Network
class GameACFFNetwork(GameACNetwork):
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device="/cpu:0"):
    GameACNetwork.__init__(self, action_size, thread_index, device)

    scope_name = "net_" + str(self._thread_index)
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      """
      self.W_conv1, self.b_conv1 = self._conv_variable([8, 8, 4, 16])  # stride=4
      self.W_conv2, self.b_conv2 = self._conv_variable([4, 4, 16, 32]) # stride=2
      self.W_fc1, self.b_fc1 = self._fc_variable([2592, 256])
      # weight for policy output layer
      self.W_fc2, self.b_fc2 = self._fc_variable([256, action_size])
      # weight for value output layer
      self.W_fc3, self.b_fc3 = self._fc_variable([256, 1])
      """
      self.W_fc1, self.b_fc1 = self._fc_variable([STATE_SIZE*4, 200])
      # weight for policy output layer
      self.W_fc2, self.b_fc2 = self._fc_variable([200, action_size])
      self.W_fc3, self.b_fc3 = self._fc_variable([200, action_size])
      # weight for value output layer
      self.W_fc4, self.b_fc4 = self._fc_variable([200, 1])

      # state (input)
      self.s = tf.placeholder("float", [None, STATE_SIZE, 4])
      s2 = tf.reshape(self.s,[-1, STATE_SIZE*4]);
   
      """
      h_conv1 = tf.nn.relu(self._conv2d(self.s,  self.W_conv1, 4) + self.b_conv1)
      h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

      h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
      h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)
      """
      self.h_fc1 = h_fc1 = tf.nn.relu(tf.matmul(s2, self.W_fc1) + self.b_fc1)
      # policy (output)
      self.mu = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2
      self.sigma = tf.matmul(h_fc1, self.W_fc3) + self.b_fc3
      # value (output)
      v_ = tf.matmul(h_fc1, self.W_fc4) + self.b_fc4
      self.v = tf.reshape( v_, [-1] )
      #SoftPlus operation
      self.sigma=tf.log(1 + tf.exp(self.sigma));
      eps=tf.random_normal([action_size]);
      #self.action=self.mu+tf.sqrt(self.sigma)*eps;
      self.action=self.mu;
      a = tf.exp(-1*(self.action-self.mu)**2/(self.sigma+1));
      b = 1/tf.sqrt(2*self.sigma*math.pi+1)
      self.prob=a*b;

  def run_policy_and_value(self, sess, s_t):
    action_out, v_out = sess.run( [self.action, self.v], feed_dict = {self.s : [s_t]} )
    return (action_out[0], v_out[0])

  def run_policy(self, sess, s_t):
    action_out = sess.run([self.action], feed_dict = {self.s : [s_t]} )
    return action_out[0]

  def run_value(self, sess, s_t):
    v_out = sess.run( self.v, feed_dict = {self.s : [s_t]} )
    return v_out[0]

  def get_vars(self):
    #return [self.W_conv1, self.b_conv1,
    #        self.W_conv2, self.b_conv2,
    #        self.W_fc1, self.b_fc1,
    #        self.W_fc2, self.b_fc2,
    #        self.W_fc3, self.b_fc3]
    return [self.W_fc1, self.b_fc1,
            self.W_fc2, self.b_fc2,
            self.W_fc3, self.b_fc3,
            self.W_fc4, self.b_fc4]

# Actor-Critic LSTM Network
class GameACLSTMNetwork(GameACNetwork):
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device="/cpu:0" ):
    GameACNetwork.__init__(self, action_size, thread_index, device)

    # state (input)
    self.s = tf.placeholder("float", [None, STATE_SIZE, 4])
    s2 = tf.reshape(self.s,[-1, STATE_SIZE*4]);

    # place holder for LSTM unrolling time step size.
    self.pstep_size = tf.placeholder(tf.float32, [1])
    self.pinitial_lstm_state0 = tf.placeholder(tf.float32, [1, 200])
    self.pinitial_lstm_state1 = tf.placeholder(tf.float32, [1, 200])
    self.pinitial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.pinitial_lstm_state0,self.pinitial_lstm_state1)
    scope_name = "net_" + str(self._thread_index)+"_policy"
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      ### policy weights
      self.pW_fc1, self.pb_fc1 = self._fc_variable([STATE_SIZE*4, 200])
      # lstm
      self.plstm = tf.contrib.rnn.BasicLSTMCell(200, state_is_tuple=True)
      # weight for policy output layer
      self.pW_fc2, self.pb_fc2 = self._fc_variable([200, action_size])
      self.pW_fc3, self.pb_fc3 = self._fc_variable([200, action_size])

      ### policy networks
      self.ph_fc1 = ph_fc1 = tf.nn.relu(tf.matmul(s2, self.pW_fc1) + self.pb_fc1)
      ph_fc1_reshaped = tf.reshape(ph_fc1, [1,-1,200])
      plstm_outputs, self.plstm_state = tf.nn.dynamic_rnn(self.plstm,
                                                        ph_fc1_reshaped,
                                                        initial_state = self.pinitial_lstm_state,
                                                        sequence_length = self.pstep_size,
                                                        time_major = False,
                                                        scope = scope)
      plstm_outputs = tf.reshape(plstm_outputs, [-1,200])
      self.mu = tf.matmul(plstm_outputs, self.pW_fc2) + self.pb_fc2
      self.sigma = tf.matmul(plstm_outputs, self.pW_fc3) + self.pb_fc3
      #SoftPlus operation
      self.sigma=tf.log(1 + tf.exp(self.sigma));
      eps=tf.random_normal([action_size]);
      self.action=self.mu+tf.sqrt(self.sigma)*eps;
      a = tf.exp(-1*(self.action-self.mu)**2/(self.sigma));
      b = 1/tf.sqrt(2*self.sigma*math.pi)
      self.prob=a*b;
      scope.reuse_variables()
      self.pW_lstm = tf.get_variable("basic_lstm_cell/kernel")
      self.pb_lstm = tf.get_variable("basic_lstm_cell/bias")
      self.reset_state()

    # place holder for LSTM unrolling time step size.
    self.vstep_size = tf.placeholder(tf.float32, [1])
    self.vinitial_lstm_state0 = tf.placeholder(tf.float32, [1, 200])
    self.vinitial_lstm_state1 = tf.placeholder(tf.float32, [1, 200])
    self.vinitial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.vinitial_lstm_state0,self.vinitial_lstm_state1)
    scope_name = "net_" + str(self._thread_index)+"_value"
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      ### value weights
      self.vW_fc1, self.vb_fc1 = self._fc_variable([STATE_SIZE*4, 200])
      # lstm
      self.vlstm = tf.contrib.rnn.BasicLSTMCell(200, state_is_tuple=True)
      # weight for value output layer
      self.vW_fc2, self.vb_fc2 = self._fc_variable([200, 1])
      ### value networks
      self.vh_fc1 = vh_fc1 = tf.nn.relu(tf.matmul(s2, self.vW_fc1) + self.vb_fc1)
      vh_fc1_reshaped = tf.reshape(vh_fc1, [1,-1,200])
      vlstm_outputs, self.vlstm_state = tf.nn.dynamic_rnn(self.vlstm,
                                                        vh_fc1_reshaped,
                                                        initial_state = self.vinitial_lstm_state,
                                                        sequence_length = self.vstep_size,
                                                        time_major = False,
                                                        scope = scope)
      vlstm_outputs = tf.reshape(vlstm_outputs, [-1,200])
      v_ = tf.matmul(vlstm_outputs, self.vW_fc2) + self.vb_fc2
      self.v = tf.reshape( v_, [-1] )
      scope.reuse_variables()
      self.vW_lstm = tf.get_variable("basic_lstm_cell/kernel")
      self.vb_lstm = tf.get_variable("basic_lstm_cell/bias")
      self.reset_state()

  def reset_state(self):
    self.plstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, 200]),
                                                        np.zeros([1, 200]))
    self.vlstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, 200]),
                                                        np.zeros([1, 200]))

  def run_policy_and_value(self, sess, s_t):
    # This run_policy_and_value() is used when forward propagating.
    # so the step size is 1.
    action_out, v_out = sess.run( [self.action, self.v],
                                                   feed_dict = {self.s : [s_t],
                                                                self.pinitial_lstm_state0 : self.plstm_state_out[0],
                                                                self.pinitial_lstm_state1 : self.plstm_state_out[1],
                                                                self.pstep_size : [1],
                                                                self.vinitial_lstm_state0 : self.vlstm_state_out[0],
                                                                self.vinitial_lstm_state1 : self.vlstm_state_out[1],
                                                                self.vstep_size : [1]} )
    # pi_out: (1,3), v_out: (1)
    return (action_out[0], v_out[0])

  def run_policy(self, sess, s_t):
    # This run_policy() is used for displaying the result with display tool.    
    action_out  = sess.run( self.action,
                                            feed_dict = {self.s : [s_t],
                                                         self.pinitial_lstm_state0 : self.plstm_state_out[0],
                                                         self.pinitial_lstm_state1 : self.plstm_state_out[1],
                                                         self.pstep_size : [1],
                                                         self.vinitial_lstm_state0 : self.vlstm_state_out[0],
                                                         self.vinitial_lstm_state1 : self.vlstm_state_out[1],
                                                         self.vstep_size : [1]} )
                                            
    return action_out[0]

  def run_value(self, sess, s_t):
    # This run_value() is used for calculating V for bootstrapping at the 
    # end of LOCAL_T_MAX time step sequence.
    # When next sequcen starts, V will be calculated again with the same state using updated network weights,
    # so we don't update LSTM state here.
    vprev_lstm_state_out = self.vlstm_state_out
    v_out = sess.run(self.v,
                         feed_dict = {self.s : [s_t],
                                      self.pinitial_lstm_state0 : self.plstm_state_out[0],
                                      self.pinitial_lstm_state1 : self.plstm_state_out[1],
                                      self.pstep_size : [1],
                                      self.vinitial_lstm_state0 : self.vlstm_state_out[0],
                                      self.vinitial_lstm_state1 : self.vlstm_state_out[1],
                                      self.vstep_size : [1]} )
    
    # roll back lstm state
    self.vlstm_state_out = vprev_lstm_state_out
    return v_out[0]

  def get_vars(self):
    return [self.pW_fc1, self.pb_fc1,
            self.pW_lstm, self.pb_lstm,
            self.pW_fc2, self.pb_fc2,
            self.pW_fc3, self.pb_fc3,
            self.vW_fc1, self.vb_fc1,
            self.vW_lstm, self.vb_lstm,
            self.vW_fc2, self.vb_fc2]
