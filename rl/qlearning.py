import math
import random

import torch
import torch.nn.functional as F
from commons.torch_utils import copynet
from .utils import DotDict

epsilon_start = 1
epsilon_end = 0.01
epsilon_decay = 5000


def epsilon_by_frame(frame):
    epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1 * frame / epsilon_decay)


class DQNAgent():
    def __init__(self, config):
        config = DotDict(**config)
        self.env = config.env
        self.nn = config.nn
        self.target_nn = copynet(self.nn)
        self.target_nn.load_state_dict(self.nn.state_dict())
        self.optim = config.optim
        self._config = config

        self.frame = 0
        self.episode = 0

        self.ep_rewards = [0]

        self._state = None

    def step(self):
        state = self.env.env.state
        self._state = torch.Tensor(state)
        qval = self.nn(self._state).detach()
        action = self.env.action_space.sample() if random.random() < epsilon_by_frame(
            self.frame) or self.frame < self._config.exploration_steps else torch.argmax(qval).item()
        next_state, reward, done, info = self.env.step(action)
        #         print(done)
        self.ep_rewards[-1] += 1
        self._config.replay_buffer.push([torch.Tensor(state),
                                         torch.tensor(action),
                                         torch.Tensor(next_state),
                                         torch.tensor(reward),
                                         torch.tensor(done)])
        if done:
            print(f'episode {self.episode}, reward {self.ep_rewards[-1]}')
            self.ep_rewards.append(0)
            self.episode += 1
            self.env.reset()

    def learn(self):
        if self.frame < self._config.exploration_steps:
            return

        states, actions, next_states, rewards, dones = self._config.replay_buffer.sample(self._config.batch_size)

        qval = self.nn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            exp_qval = self.target_nn(next_states).max(1)[0]
        #         print(exp_qval)
        #         print(self._config.gamma)
        target = rewards + self._config.gamma * exp_qval * (1 - dones.float())

        loss = F.mse_loss(qval, target)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def run(self, n_frames, cur_frame=0):
        self.frame = cur_frame
        self._state = self.env.reset()
        for _ in range(n_frames):
            self.step()
            self.frame += 1
            self.learn()
            if self.frame % self._config.update_target == 0:
                self.target_nn.load_state_dict(self.nn.state_dict())

            if len(self.ep_rewards) >= 100 and sum(self.ep_rewards) / len(self.ep_rewards) >= 195:
                print('Solved')
                break
