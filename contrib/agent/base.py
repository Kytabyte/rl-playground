"""
Base Agent
"""

from ..util import DotDict
from ..env import Env


class Base:
    def __init__(self, config):
        self._config = DotDict(**config)
        self._stat = DotDict(**{
            'frame': 0,
            'episode': 0,
            'rewards': [0]
        })
        env_name = self._config.env
        self._config.env = Env(env_name, self._stat)

    @property
    def env(self):
        return self._config.env

    @property
    def nn(self):
        return self._config.nn

    @property
    def optim(self):
        return self._config.optim

    @property
    def scheduler(self):
        return self._config.scheduler

    @property
    def stat(self):
        return self._stat

    @property
    def frame(self):
        return self._stat.frame

    @property
    def episode(self):
        return self._stat.episode

    def step(self):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def run(self, n_frames, from_frame=0):
        self._stat.frame = from_frame

        while self.frame <= n_frames:
            for _ in range(self._config.step_length):
                self.step()
                self._stat.frame += 1
            self.learn()

            if len(self._stat.rewards) >= 100 and sum(self._stat.rewards) / len(self._stat.rewards) >= 195:
                print('Solved')
                break
