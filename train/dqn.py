from models import MLP
from envs import Env2048
from rl import QNet

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Constants

env = Env2048()
n_obs, n_act = env.observation_space.shape[0], env.action_space.n

net = MLP(n_obs, n_act, hidden=(128, 128))
qnet = QNet(net, (n_obs,), n_act, target=True)

print(qnet._net)

replay_size = 5000
batch_size = 256

n_episode = 20
update_target_freq = 2

epsilon_start = 1
epsilon_final = 0.01
epsilon_decay = 500
eps = lambda frame_idx: epsilon_final + (
        epsilon_start - epsilon_final) * np.exp(-1 * frame_idx / epsilon_decay)

optimizer = optim.Adam(qnet.parameters())
loss_fn = nn.MSELoss()

for i_episode in range(n_episode):
    obs = env.reset()
    valid_move = None

    for t in range(2000):
        action = qnet.act(obs.reshape(1, -1), act_mask=None, eps=eps(i_episode * 200 + t))
        next_obs, reward, done, info = env.step(action)
        valid_move = info.get('valid_move', None)

        qnet.push_buffer((obs, action, next_obs, reward, int(done)))

        obs = next_obs

        qnet.learn(batch_size, optimizer, loss_fn)

        if done:
            print(obs.reshape(4, 4))
            print("Episode finished after {} timesteps".format(t + 1))
            break

    if i_episode % update_target_freq == 0:
        qnet.update_tnet()
