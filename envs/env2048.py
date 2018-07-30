import sys
import gym
from io import StringIO

from gym import spaces
from gym.utils import seeding

import numpy as np

from games import Play2048


class Env2048(gym.Env):
  def __init__(self, size=4):
    self._player = Play2048(size)
    self.action_space = spaces.Discrete(size)
    self.observation_space = spaces.Box(0, 2 ** (size * size), (size * size, ), dtype=np.int)
    self.metadata = {'render.modes': ['human', 'ansi']}
    self._score = self._player.score()
    
  def _get_obs(self):
    return self._player.status().flatten()
  
  def _get_reward(self):
    score = self._player.score()
    reward = score - self._score
    self._score = score
    return reward

  def _is_gameover(self):
    return self._player.is_terminate()

  def _info(self):
    return {'valid_move': self._player.can_move(), 'highest': np.max(self._get_obs())}

  def n_obs(self):
    return self.observation_space.shape[0] * self.observation_space.shape[1]

  def seed(self, seed=None):
    _, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    self._player.move(action)
    obs = self._get_obs()
    rwd = self._get_reward()
    done = self._is_gameover()
    info = self._info()

    return obs, rwd, done, info

  def render(self, mode='human'):
    outfile = StringIO() if mode == 'ansi' else sys.stdout
    s = str(self._player)
    outfile.write(s)
    return outfile

  def reset(self):
    self._player.reset()
    self._score = self._player.score()
    return self._get_obs()

ACTION_MEANING = {
    0: 'UP',
    1: 'RIGHT',
    2: 'DOWN',
    3: 'LEFT'
}
