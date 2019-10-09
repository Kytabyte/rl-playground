import gym
import torch


class Env:
    def __init__(self, name: str, stat: dict):
        self._name = name
        self._env = gym.make(name)
        self._stat = stat
        self._state = torch.FloatTensor(self._env.reset())

    @property
    def name(self):
        return self._name

    @property
    def state(self):
        return self._state

    def step(self, action):
        next_state, reward, done, info = self._env.step(action.item())
        self._stat.rewards[-1] += reward
        if done:
            print(self._stat.frame, self._stat.rewards[-1])
            self._stat.rewards.append(0)
        self._state = torch.FloatTensor(self._env.reset()) if done else torch.FloatTensor(next_state)
        return torch.FloatTensor(next_state), torch.tensor(reward), torch.tensor(done), info
