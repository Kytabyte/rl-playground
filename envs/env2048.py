from . import spaces
import sys

from games._2048 import Play2048


class Env2048(object):
    def __init__(self):
        self.player = Play2048()
        self._action_set = {0,1,2,3}
        self.action_space = spaces.Discrete(len(self._action_set))
        
    def _get_obs(self):
        return self.player.status()
    
    def _get_reward(self):
        return self.player.score()
    
    def _is_gameover(self):
        return self.player.is_terminate()
    
    def _info(self):
        return {'valid_move': self.player.can_move()}

    def n_obs(self):
        return self._get_obs().shape
    
    def step(self, action):
        self.player.move(action)
        obs = self._get_obs()
        rwd = self._get_reward()
        done = self._is_gameover()
        info = self._info()
        
        return obs, rwd, done, info
    
    def reset(self):
        self.player.reset()
        return self._get_obs()
    
ACTION_MEANING = {
    0: 'UP',
    1: 'RIGHT',
    2: 'DOWN',
    3: 'LEFT'
}