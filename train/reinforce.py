import torch
import torch.nn as nn
import torch.optim as optim
import gym

from rl import PolicyNet
from models import MLP

env = gym.make('2048-v0')
n_obs, n_act = env.observation_space.shape[0], env.action_space.n
n_episode = 20

model = MLP(n_input=n_obs, n_output=n_act)
net = PolicyNet(net=model,obs_shape=(n_obs, ), n_act=n_act)
optimizer = optim.Adam(net.parameters(), lr=1e-2)


from itertools import count

for i_eps in count(1):
  obs = env.reset()
  valid_move = None
  done = False
  
  ep_score = 0
  for step in count(1):
    action = net.act(obs.reshape(1,-1), act_mask=valid_move)
    obs, reward, done, info = env.step(action)
    net.rewards.append(reward * (1 - int(done)))
    valid_move = info.get('valid_move', None)

    ep_score += reward
    
    if step % 1000 == 0:
      print(obs,action, valid_move, done)
      
    if done or step > 5000:
      break
      
  print(obs)
  print('episode {} finished with step {} and score {}'.format(i_eps, step, ep_score))
  
  net.learn(optimizer, 0.99)
  obs = env.reset()
  valid_move = None
  done = False
  
  if i_eps > n_episode:
    break
