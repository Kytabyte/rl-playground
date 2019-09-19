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
import torch.distributions as distributions
import gym
from typing import Optional, Tuple

import games.gridworld as gridworld


class EnvGridWorld(gym.Env):
    """
    The environment of GridWorld from CS885 course at University of Waterloo
    """
    def __init__(self, a: float = 0.8, b: float = 0.1, terminate: Optional[int] = 100):
        self._game = gridworld.GridWorld(a, b)
        self._state = -1
        self._count = -1 if terminate is None else terminate
        self._terminate = terminate

    def reset(self) -> None:
        self._state = random.randint(0, self._game.N_STATES - 1)
        self._count = 0

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        if self._state < 0:
            raise Exception("You need to call reset first before calling step")
        if self._count > 0:
            self._count += 1

        self._state, reward = self._game.move(self._state, action)
        done = False if self._terminate is None else self._count == self._terminate

        return self._state, reward, done, {}

    def render(self, mode='human'):
        return self._state


transitions, rewards = gridworld.make_env()
