# -*- coding: utf-8 -*-
import sys
import numpy as np
import gym
from gym import wrappers
from constants import ACTION_SIZE
from constants import STATE_SIZE

ENV_NAME = 'Reacher-v1'
env = gym.make(ENV_NAME);

""" crate a new environment class with actions and states normalized to [-1,1] """
acsp = env.action_space
obsp = env.observation_space
if not type(acsp)==gym.spaces.box.Box:
  raise RuntimeError('Environment with continous action space (i.e. Box) required.')
if not type(obsp)==gym.spaces.box.Box:
  raise RuntimeError('Environment with continous observation space (i.e. Box) required.')
env_type = type(env)

class GameState(env_type):
  def __init__(self):
    self.__dict__.update(env.__dict__) # transfer properties
    # Observation space
    if np.any(obsp.high < 1e10):
      h = obsp.high
      l = obsp.low
      sc = h-l
      self.o_c = (h+l)/2.
      self.o_sc = sc / 2.
    else:
      self.o_c = np.zeros_like(obsp.high)
      self.o_sc = np.ones_like(obsp.high)
    # Action space
    h = acsp.high
    l = acsp.low
    sc = (h-l)
    self.a_c = (h+l)/2.
    self.a_sc = sc / 2.
    # Rewards
    self.r_sc = 0.1
    self.r_c = 0.
    # Special cases
    if ENV_NAME == "Reacher-v1":
      self.o_sc[6] = 40.
      self.o_sc[7] = 20.
      self.r_sc = 200.
      self.r_c = 0.
    # Check and assign transformed spaces
    self.observation_space = gym.spaces.Box(self.filter_observation(obsp.low),self.filter_observation(obsp.high))
    self.action_space = gym.spaces.Box(-np.ones_like(acsp.high),np.ones_like(acsp.high))
    def assertEqual(a,b): assert np.all(a == b), "{} != {}".format(a,b)
    assertEqual(self.filter_action(self.action_space.low), acsp.low)
    assertEqual(self.filter_action(self.action_space.high), acsp.high)
  
  def filter_observation(self,obs):
    return (obs-self.o_c) / self.o_sc
  def filter_action(self,action):
    return self.a_sc*action+self.a_c
  def filter_reward(self,reward):
    ''' has to be applied manually otherwise it makes the reward_threshold invalid '''
    return self.r_sc*reward+self.r_c

  def process(self,action):
    ac_f = np.clip(self.filter_action(action),self.action_space.low,self.action_space.high)
    obs, reward, term, info = env_type.step(self,ac_f) # super function
    reward=self.filter_reward(reward);
    obs_f = self.filter_observation(obs)
    #return obs_f, reward, term, info
    self.reward = reward;
    self.terminal = term
    self.s_t1=np.append(self.s_t[:,1:],np.reshape(obs_f,[STATE_SIZE,1]),axis=1);
  
  def reset_gs(self,x_t):
    self.reward = 0
    self.terminal = False
    self.s_t = np.stack((x_t, x_t, x_t, x_t), axis = 1)
  
  def update(self):
    self.s_t = self.s_t1
  """
  def _process_frame(self, action, reshape):
    reward = self.ale.act(action)
    terminal = self.ale.game_over()
    # screen shape is (210, 160, 1)
    self.ale.getScreenGrayscale(self._screen)
    # reshape it into (210, 160)
    reshaped_screen = np.reshape(self._screen, (210, 160))
    # resize to height=110, width=84
    resized_screen = cv2.resize(reshaped_screen, (84, 110))
    x_t = resized_screen[18:102,:]
    if reshape:
      x_t = np.reshape(x_t, (84, 84, 1))
    x_t = x_t.astype(np.float32)
    x_t *= (1.0/255.0)
    return reward, terminal, x_t
    
  def process(self, action):
    # convert original 18 action index to minimal action set index
    real_action = self.real_actions[action]
    
    r, t, x_t1 = self._process_frame(real_action, True)

    self.reward = r
    self.terminal = t
    self.s_t1 = np.append(self.s_t[:,:,1:], x_t1, axis = 2)    
  """
