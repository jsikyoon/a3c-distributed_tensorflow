# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import time
import sys

from game_state import GameState
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork

from constants import GAMMA
from constants import ENTROPY_BETA
from constants import USE_LSTM

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000

class A3CTrainingThread(object):
  def __init__(self,
               thread_index,
               global_network,
               pinitial_learning_rate,
               plearning_rate_input,
               pgrad_applier,
               vinitial_learning_rate,
               vlearning_rate_input,
               vgrad_applier,
               max_global_time_step,
               device,task_index=""):

    self.thread_index = thread_index
    self.plearning_rate_input = plearning_rate_input
    self.vlearning_rate_input = vlearning_rate_input
    self.max_global_time_step = max_global_time_step
    self.game_state = GameState()
    state=self.game_state.reset();
    self.game_state.reset_gs(state);
    self.action_size=self.game_state.action_size;
    self.state_size=self.game_state.state_size;
    self.local_max_iter=self.game_state.local_max_iter;

    if USE_LSTM:
      self.local_network = GameACLSTMNetwork(self.action_size,self.state_size,self.game_state.action_low,self.game_state.action_high, thread_index, device)
    else:
      self.local_network = GameACFFNetwork(self.action_size,self.state_size,self.game_state.action_low,self.game_state.action_high, thread_index, device)

    self.local_network.prepare_loss(ENTROPY_BETA)

    with tf.device(device):
      pvar_refs = [v._ref() for v in self.local_network.get_pvars()]
      self.policy_gradients = tf.gradients(
        self.local_network.policy_loss, pvar_refs,
        gate_gradients=False,
        aggregation_method=None,
        colocate_gradients_with_ops=False)
      vvar_refs = [v._ref() for v in self.local_network.get_vvars()]
      self.value_gradients = tf.gradients(
        self.local_network.value_loss, vvar_refs,
        gate_gradients=False,
        aggregation_method=None,
        colocate_gradients_with_ops=False)

    self.apply_policy_gradients = pgrad_applier.apply_gradients(
      self.local_network.get_pvars(),
      self.policy_gradients )
    self.apply_value_gradients = vgrad_applier.apply_gradients(
      self.local_network.get_vvars(),
      self.value_gradients )
    
    self.local_t = 0

    self.pinitial_learning_rate = pinitial_learning_rate
    self.vinitial_learning_rate = vinitial_learning_rate

    self.episode_reward = 0

    # variable controling log output
    self.prev_local_t = 0

  def _panneal_learning_rate(self, global_time_step):
    learning_rate = self.pinitial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
    if learning_rate < 0.0:
      learning_rate = 0.0
    return learning_rate
  
  def _vanneal_learning_rate(self, global_time_step):
    learning_rate = self.vinitial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
    if learning_rate < 0.0:
      learning_rate = 0.0
    return learning_rate

  def choose_action(self, pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)

  def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t):
    summary_str = sess.run(summary_op, feed_dict={
      score_input: score
    })
    summary_writer.add_summary(summary_str, global_t)
    summary_writer.flush()
    
  def set_start_time(self, start_time):
    self.start_time = start_time

  def process(self, sess, global_t, summary_writer, summary_op, score_input,score_ph="",score_ops=""):
    states = []
    actions = []
    rewards = []
    values = []

    terminal_end = False

    start_local_t = self.local_t

    if USE_LSTM:
      pstart_lstm_state = self.local_network.plstm_state_out
      vstart_lstm_state = self.local_network.vlstm_state_out

    # t_max times loop
    for i in range(self.local_max_iter):
      action, value_ = self.local_network.run_policy_and_value(sess, self.game_state.s_t)
      states.append(self.game_state.s_t)
      actions.append(action)
      values.append(value_)

      # process game
      self.game_state.process(action)

      # receive game result
      reward = self.game_state.reward
      terminal = self.game_state.terminal

      self.episode_reward += reward

      # clip reward
      #rewards.append( np.clip(reward,-1,1) )
      rewards.append(reward);

      self.local_t += 1

      # s_t1 -> s_t
      self.game_state.update()
      if terminal:
        terminal_end = True
        print("score={}".format(self.episode_reward/self.game_state.r_sc))
        score=self.episode_reward/self.game_state.r_sc;
        if summary_writer:
          self._record_score(sess, summary_writer, summary_op, score_input,
            self.episode_reward/self.game_state.r_sc, global_t)
        else:
          sess.run(score_ops,{score_ph:self.episode_reward/self.game_state.r_sc});

        self.episode_reward = 0
        state=self.game_state.reset()
        self.game_state.reset_gs(state);
        if USE_LSTM:
          self.local_network.reset_state()
        break

    R = 0.0
    if not terminal_end:
      R = self.local_network.run_value(sess, self.game_state.s_t)
      score=self.episode_reward/self.game_state.r_sc;

    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()

    batch_si = []
    batch_a = []
    batch_td = []
    batch_R = []

    # compute and accmulate gradients
    for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
      R = ri + GAMMA * R
      td = R - Vi

      batch_si.append(si)
      batch_R.append(R)
      batch_td.append(td);

    pcur_learning_rate = self._panneal_learning_rate(global_t)
    vcur_learning_rate = self._vanneal_learning_rate(global_t)

    if USE_LSTM:
      batch_si.reverse()
      batch_td.reverse()
      batch_R.reverse()
      
      sess.run( self.apply_policy_gradients,
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.td: batch_td,
                  self.local_network.r: batch_R,
                  self.local_network.pinitial_lstm_state: pstart_lstm_state,
                  self.local_network.pstep_size : [len(batch_a)],
                  self.local_network.vinitial_lstm_state: vstart_lstm_state,
                  self.local_network.vstep_size : [len(batch_a)],
                  self.plearning_rate_input: pcur_learning_rate } )
      sess.run( self.apply_value_gradients,
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.td: batch_td,
                  self.local_network.r: batch_R,
                  self.local_network.pinitial_lstm_state: pstart_lstm_state,
                  self.local_network.pstep_size : [len(batch_a)],
                  self.local_network.vinitial_lstm_state: vstart_lstm_state,
                  self.local_network.vstep_size : [len(batch_a)],
                  self.vlearning_rate_input: vcur_learning_rate } )
    else:
      sess.run( self.apply_policy_gradients,
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.r: batch_R,
                  self.local_network.td: batch_td,
                  self.plearning_rate_input: pcur_learning_rate} )
      sess.run( self.apply_value_gradients,
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.r: batch_R,
                  self.local_network.td: batch_td,
                  self.vlearning_rate_input: vcur_learning_rate} )
      
    if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
      self.prev_local_t += PERFORMANCE_LOG_INTERVAL
      elapsed_time = time.time() - self.start_time
      steps_per_sec = global_t / elapsed_time
      #print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
      #  global_t,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

    # return advanced local step size
    diff_local_t = self.local_t - start_local_t
    return diff_local_t
    
