import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from commons.torch_utils import astensor, copynet
import commons.constants as const


class PolicyNet(object):
  """
    REINFORCE algorithm implementation
  """
  def __init__(self, net, obs_shape, n_act, target=False):
    self._net = net
    self._obs_shape, self._n_act = obs_shape, n_act
    self._tnet = copynet(self._net) if target else None
    
    self.logprobs = []
    self.rewards = []
  
  def forward(self, obs):
    if not isinstance(obs, torch.Tensor):
      obs = astensor(obs, 'float')

    obs.reshape(obs.size(0), *self._obs_shape)
    
    if self._tnet:
      return F.softmax(self._tnet(obs), dim=1)

    return F.softmax(self._net(obs), dim=1)
  
  def parameters(self):
    return self._net.parameters()

  def learn(self, optimizer=None, gamma=0.95):
    assert len(self.logprobs) > 0 and len(self.rewards) > 0
    assert len(self.logprobs) == len(self.rewards)
    
    mean, std = astensor(self.rewards,'float').mean().item(), astensor(self.rewards,'float').std().item()
    
    gain = 0
    loss = []
    while len(self.rewards) > 0:
      logprob, reward = self.logprobs.pop(), (self.rewards.pop() - mean) / (std + const.MACHINE_EPS)
      gain = reward + gamma * gain
      loss.append(-gain * logprob)
    
    if optimizer is None:
      optimizer = optim.Adam(self.parameters())

    optimizer.zero_grad()
    loss = torch.cat(loss).sum()
    loss.backward()
    optimizer.step()
    
  
  def act(self, obs, act_mask=None):
    act_mask = astensor([1] * self._n_act, 'float') if act_mask is None else astensor(act_mask, 'float')
    
    probs = self.forward(obs)
    # print(probs)
    action = (probs * act_mask.unsqueeze(0)).argmax().item()
    
    self.logprobs.append(torch.log(probs))
    return action

  def update_tnet(self, soft_tau=1e-2):
    if self._tnet:
      for target_param, param in zip(self._tnet.parameters(), self._net.parameters()):
        target_param.data.copy_(target_param.data * (1 - soft_tau) + param.data * soft_tau)
    else:
      print('You did not initialize target network, please check your network structure.')