from functools import reduce

from networks.mlp import MLP
from envs.env2048 import Env2048
from rl.qlearning import QNet
from rl.utils import ReplayBuffer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

#### Constants

env = Env2048()
n_obs, n_act = reduce(lambda x, y: x * y, env.n_obs()), len(env.action_space)

net = MLP(n_obs, n_act, hidden=(128,128))
qnet = QNet(net, n_obs, n_act, target=True)

print(qnet._net)

replay_size = 5000
batch_size = 256

replay_buffer = ReplayBuffer(replay_size)

n_episode = 20
update_target_freq=2

epsilon_start = 1
epsilon_final = 0.01
epsilon_decay = 500
eps = lambda frame_idx: epsilon_final + (
  epsilon_start -epsilon_final) * np.exp(-1 * frame_idx / epsilon_decay)

optimizer = optim.Adam(qnet.parameters())
loss_fn = nn.MSELoss()


for i_episode in range(n_episode):
    obs = env.reset()
    valid_move = None

    for t in range(2000):
        if t % 50 == 0:
            print(obs)

        action = qnet.act(obs.flatten(), act_mask=valid_move, eps=eps(i_episode * 200 + t))
        next_obs, reward, done, info = env.step(action)
        valid_move = info.get('valid_move', None)

        replay_buffer.push((obs, action, next_obs, reward, int(done)))
            
        obs = next_obs
        
        if len(replay_buffer) > batch_size:
            obses, actions, next_obses, rewards, dones = tuple(replay_buffer.sample(batch_size))
            obses = np.array(obses).reshape((batch_size, -1))
            next_obses = np.array(next_obses).reshape((batch_size, -1))
            qnet.learn((obses, actions, next_obses, rewards, dones), optimizer, loss_fn)

        if done:
            print(obs)
            print("Episode finished after {} timesteps".format(t+1))
            break

    if i_episode % update_target_freq == 0:
        qnet.update_tnet()

