import random

import torch
import torch.nn as nn
import torch.optim as optim

from commons.torch_utils import astensor, copynet
from .utils import ReplayBuffer

class QNet(object):
  def __init__(self, net, obs_shape, n_act, buffer_size=1000, target=False):
    self._net = net
    if target:
      self._tnet = copynet(self._net)
      self._tnet.load_state_dict(self._net.state_dict())
    else:
      self._tnet = None

    self._obs_shape, self._n_act = obs_shape, n_act
    self.replay_buffer = ReplayBuffer(buffer_size)

  def push_buffer(self, sample):
    assert len(sample) == 5
    self.replay_buffer.push(sample)

  def parameters(self):
    return self._net.parameters()

  def forward(self, obs, tnet=True):
    if not isinstance(obs, torch.Tensor):
      obs = astensor(obs, 'float')

    obs = obs.reshape(obs.size(0), *self._obs_shape)

    if self._tnet and tnet:
      return self._tnet(obs)
    return self._net(obs)

  def learn(self, batch_size, optimizer=None, loss_fn=None, gamma=0.95):
    if len(self.replay_buffer) < batch_size:
      return

    obs, act, next_obs, reward, done = self.replay_buffer.sample(batch_size)

    obs = astensor(obs, 'float')
    act = astensor(act, 'long')
    next_obs = astensor(next_obs, 'float')
    reward = astensor(reward, 'float')
    done = astensor(done, 'float')

    qval = self.forward(obs, tnet=False)
    with torch.no_grad():
      opt_qval = self.forward(next_obs)

    qval = qval.gather(1, act.unsqueeze(1)).squeeze(1)
    opt_qval = opt_qval.max(1)[0]
    exp_qval = reward + gamma * opt_qval * (1 - done)

    if optimizer is None:
      optimizer = optim.Adam(self.parameters())

    if loss_fn is None:
      loss_fn = nn.MSELoss()

    if optimizer is None:
      optimizer = optim.Adam(self.parameters())

    loss = loss_fn(qval, exp_qval)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
  def act(self, obs, act_mask=None, eps=0.2):
    act_mask = astensor(act_mask) if act_mask is not None else astensor([1] * self._n_act, 'byte')
    
    if random.random() < eps:
      valid_move = astensor(range(self._n_act), 'int').masked_select(act_mask)
      action = valid_move[random.randrange(len(valid_move))].item()
    else:
      with torch.no_grad():
        val = self.forward(obs)

        # This is quite dangerous but we know what we are doing here
        maxval = val.masked_select(act_mask.unsqueeze(0)).argmax()
        action = maxval.item()
    
    return action
  
  def update_tnet(self, soft_tau=1e-2):
    if self._tnet:
      for target_param, param in zip(self._tnet.parameters(), self._net.parameters()):
        target_param.data.copy_(target_param.data * (1 - soft_tau) + param.data * soft_tau)
    else:
      print('You did not initialize target network, please check your network structure.')