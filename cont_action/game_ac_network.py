# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import math

# Actor-Critic Network Base Class
# (Policy network and Value network)
class GameACNetwork(object):
  def __init__(self,
               action_size,
               state_size,
               thread_index, # -1 for global               
               device="/cpu:0"):
    self._action_size = action_size
    self._state_size = state_size
    self._thread_index = thread_index
    self._device = device    

  def prepare_loss(self, entropy_beta):
    with tf.device(self._device):
    
      # temporary difference (R-V) (input for policy)
      self.td = tf.placeholder("float", [None])

      # policy loss
      self.policy_loss=policy_loss = -1*self.log_prob*self.td -1e-1*self.entropy;

      # R (input for value)
      self.r = tf.placeholder("float", [None])
      
      # value loss (output)
      # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
      self.value_loss=value_loss = tf.squared_difference(self.r, self.v)

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
    #weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
    #bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
    weight = tf.Variable(tf.random_uniform(weight_shape))
    bias   = tf.Variable(tf.random_uniform(bias_shape))
    return weight, bias

# Actor-Critic FF Network
class GameACFFNetwork(GameACNetwork):
  def __init__(self,
               action_size,
               state_size,
               action_low,
               action_high,
               thread_index, # -1 for global
               device="/cpu:0"):
    GameACNetwork.__init__(self, action_size, state_size,thread_index, device)

    # state (input)
    self.s = tf.placeholder("float", [None, state_size, 4])
    s2 = tf.reshape(self.s,[-1, state_size*4]);

    scope_name = "net_" + str(self._thread_index) + "_policy"
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      # for mu
      self.pW_fc1, self.pb_fc1 = self._fc_variable([state_size*4, action_size])
      # for sigma
      self.pW_fc2, self.pb_fc2 = self._fc_variable([state_size*4, action_size])
      # policy (output)
      self.mu = tf.matmul(s2, self.pW_fc1) + self.pb_fc1
      self.sigma = tf.matmul(s2, self.pW_fc2) + self.pb_fc2
      self.sigma = tf.nn.softplus(self.sigma)
      #SoftPlus operation
      self.normal_dist=tf.contrib.distributions.Normal(self.mu,self.sigma);
      self.action = self.normal_dist._sample_n(1)
      self.action = tf.clip_by_value(self.action,action_low,action_high);
      self.log_prob = self.normal_dist.log_prob(self.action);
      self.entropy = self.normal_dist.entropy();

    scope_name = "net_" + str(self._thread_index) + "_value"
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      # for value
      self.vW_fc1, self.vb_fc1 = self._fc_variable([state_size*4, action_size])
      v_ = tf.matmul(s2,self.vW_fc1)+self.vb_fc1;
      self.v = tf.reshape(v_,[-1]);

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
    return [self.pW_fc1, self.pb_fc1,
            self.pW_fc2, self.pb_fc2,
            self.vW_fc1, self.vb_fc1]
  def get_pvars(self):
    return [self.pW_fc1, self.pb_fc1,
            self.pW_fc2, self.pb_fc2]
  def get_vvars(self):
    return [self.vW_fc1, self.vb_fc1]

# Actor-Critic LSTM Network
class GameACLSTMNetwork(GameACNetwork):
  def __init__(self,
               action_size,
               state_size,
               action_low,
               action_high,
               thread_index, # -1 for global
               device="/cpu:0" ):
    GameACNetwork.__init__(self, action_size,state_size, thread_index, device)

    # state (input)
    self.s = tf.placeholder("float", [None, state_size, 4])
    s2 = tf.reshape(self.s,[-1, state_size*4]);
    

    # place holder for LSTM unrolling time step size.
    self.pstep_size = tf.placeholder(tf.float32, [1])
    self.pinitial_lstm_state0 = tf.placeholder(tf.float32, [1, 200])
    self.pinitial_lstm_state1 = tf.placeholder(tf.float32, [1, 200])
    self.pinitial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.pinitial_lstm_state0,self.pinitial_lstm_state1)
    scope_name = "net_" + str(self._thread_index)+"_policy"
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      ### policy weight
      
      self.pW_fc1, self.pb_fc1 = self._fc_variable([state_size*4, 200])
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
      self.sigma = tf.nn.softplus(self.sigma)+1e-5
      #SoftPlus operation
      self.normal_dist=tf.contrib.distributions.Normal(self.mu,self.sigma);
      self.action = self.normal_dist._sample_n(1)
      self.action = tf.clip_by_value(self.action,action_low,action_high);
      self.log_prob = self.normal_dist.log_prob(self.action);
      self.entropy = self.normal_dist.entropy();
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
      self.vW_fc1, self.vb_fc1 = self._fc_variable([state_size*4, 200])
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
  
  def get_pvars(self):
    return [self.pW_fc1, self.pb_fc1,
            self.pW_lstm, self.pb_lstm,
            self.pW_fc2, self.pb_fc2,
            self.pW_fc3, self.pb_fc3]
  
  def get_vvars(self):
    return [self.vW_fc1, self.vb_fc1,
            self.vW_lstm, self.vb_lstm,
            self.vW_fc2, self.vb_fc2]
