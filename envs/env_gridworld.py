"""
Construct a simple maze MDP

  Grid world layout:

  ---------------------
  |  0 |  1 |  2 |  3 |
  ---------------------
  |  4 |  5 |  6 |  7 |
  ---------------------
  |  8 |  9 | 10 | 11 |
  ---------------------
  | 12 | 13 | 14 | 15 |
  ---------------------

  Goal state: 15
  Bad state: 9
  End state: 16

  The end state is an absorbing state that the agent transitions
  to after visiting the goal state.

  There are 17 states in total (including the end state)
  and 4 actions (up, down, left, right).
"""

import random
import gym
from typing import Tuple

import games.gridworld as gridworld


class EnvGridWorld(gym.Env):
    """
    The environment of GridWorld from CS885 course at University of Waterloo
    """
    def __init__(self, a: float = 0.8, b: float = 0.1):
        self._game = gridworld.GridWorld(a, b)
        self._state = -1
        self._terminate = False

    def reset(self) -> None:
        self._state = random.randint(0, self._game.N_STATES - 1)

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        if self._state < 0:
            raise Exception("You need to call reset first before calling step")
        if self._terminate:
            raise Exception("You need to call reset because the game is terminated")

        self._state, reward, done = self._game.move(self._state, action)
        return self._state, reward, done, {}

    def render(self, mode='human'):
        return self._state


transitions, rewards = gridworld.make_env()
