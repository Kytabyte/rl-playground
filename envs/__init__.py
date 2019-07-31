from envs.env2048 import Env2048
from envs.env import ObservableEnv
from gym.envs.registration import register
from . import env_gridworld

__all__ = ['Env2048', 'ObservableEnv', 'env_gridworld']

register(
    id='2048-v0',
    entry_point='envs.env2048:Env2048',
    kwargs={'size': 4}
)