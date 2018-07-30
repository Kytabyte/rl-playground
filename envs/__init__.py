from envs.env2048 import *
from gym.envs.registration import register

__all__ = ['Env2048']

register(
    id='2048-v0',
    entry_point='envs.env2048:Env2048',
    kwargs={'size': 4}
)