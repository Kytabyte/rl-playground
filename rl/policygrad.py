import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from commons.torch_utils import astensor, copynet
import commons.constants as const

from . import QNet


class PolicyNet(object):
  """
    REINFORCE algorithm implementation
  """

  def __init__(self, net, obs_shape, n_act, target=False):
    self._net = net
    self._obs_shape, self._n_act = obs_shape, n_act
    if target:
      self._tnet = copynet(self._net)
      self._tnet.load_state_dict(self._net.state_dict())
    else:
      self._tnet = None

    self.logprobs = []
    self.rewards = []
    self.dones = []

  def act(self, obs, act_mask=None):
    act_mask = astensor(act_mask,
                        'float') if act_mask is not None else astensor(
                            [1] * self._n_act, 'float')

    probs = self.forward(obs) * act_mask.unsqueeze(0) + const.MACHINE_EPS
    distribution = Categorical(probs)
    action = distribution.sample()
    self.logprobs.append(distribution.log_prob(action))

    return action.cpu().item()

  def forward(self, obs):
    if not isinstance(obs, torch.Tensor):
      obs = astensor(obs, 'float')
    obs.reshape(obs.size(0), *self._obs_shape)
    if self._tnet:
      return F.softmax(self._tnet(obs), dim=1)
    return F.softmax(self._net(obs), dim=1)

  def learn(self, optimizer=None, gamma=0.99):
    assert self.logprobs and self.rewards and self.dones
    assert len(self.logprobs) == len(self.rewards) == len(self.dones)

    if optimizer is None:
      optimizer = optim.Adam(self.parameters())

    discounted_rewards = self._discount(gamma)
    loss = self._loss(discounted_rewards)
    self._learn(optimizer, loss)

  def parameters(self):
    return self._net.parameters()

  def update_tnet(self, soft_tau=1e-2):
    if self._tnet:
      for target_param, param in zip(self._tnet.parameters(),
                                     self._net.parameters()):
        target_param.data.copy_(target_param.data * (1 - soft_tau) +
                                param.data * soft_tau)
    else:
      print(
          'You did not initialize target network, please check your network structure.'
      )

  def _discount(self, gamma, init_gain=0):
    gain = init_gain
    discounted_rewards = []
    while self.rewards:
      reward, done = self.rewards.pop(), self.dones.pop()
      gain = reward + gamma * gain * (1 - done)
      discounted_rewards.append(gain)

    with torch.no_grad():
      discounted_rewards = astensor(discounted_rewards)
      discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
          discounted_rewards.std() + const.MACHINE_EPS)

    return discounted_rewards

  def _loss(self, rewards):
    loss = []
    for reward in rewards:
      logprob = self.logprobs.pop()
      loss.append(-logprob * reward)
      loss = torch.cat(loss).sum()
    return loss

  @staticmethod
  def _learn(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


class ActorCritic(PolicyNet):
  """
    an implementation of Actor-Critic algorithm
  """

  def __init__(self, actor_net, critic_net, obs_shape, n_act, target=False):
    super(ActorCritic, self).__init__(actor_net, obs_shape, n_act, target)
    self._value_net = QNet(critic_net, obs_shape, n_act, target=target)
    self.values = []

  def actor(self, obs):
    return self.forward(obs)

  def critic(self, obs):
    return self._value_net.forward(obs)

  def learn(self, optimizer=None, gamma=0.99):
    assert self.logprobs and self.rewards and self.values
    assert len(self.logprobs) == len(self.rewards) == len(self.values)
    if optimizer is None:
      optimizer = optim.Adam(self.parameters())

    discounted_rewards = self._discount(gamma)
    loss = self._loss(discounted_rewards)
    self._learn(optimizer, loss)

  def act(self, obs, act_mask=None):
    critic = self.critic(obs)
    self.values.append(critic)
    return super(ActorCritic, self).act(obs, act_mask)

  def update_tnet(self, soft_tau=1e-2):
    super(ActorCritic, self).update_tnet(soft_tau)
    self._value_net.update_tnet(soft_tau)

  def _loss(self, rewards):
    actor_loss, critic_loss = [], []
    for reward in rewards:
      logprob, value = self.logprobs.pop(), self.values.pop()
      actor_loss.append(-logprob * (reward - value.detach().squeeze()))
      critic_loss.append(F.smooth_l1_loss(value.squeeze(), reward))
      loss = torch.stack(actor_loss).sum() + torch.stack(critic_loss).sum()
    return loss


class A2C(ActorCritic):
  """
    an implementation of Advanced Actor-Critic algorithm
  """

  def __init__(self, actor_net, critic_net, obs_shape, n_act, target=False):
    super(A2C, self).__init__(actor_net, critic_net, obs_shape, n_act, target)
    self.last_value = 0

  def learn(self, optimizer=None, gamma=0.99):
    assert self.logprobs and self.rewards and self.values
    assert len(self.logprobs) == len(self.rewards) == len(self.values) == len(
        self.dones)
    if optimizer is None:
      optimizer = optim.Adam(self.parameters())

    discounted_rewards = self._discount(gamma, init_gain=self.last_value)
    loss = self._loss(discounted_rewards)
    self._learn(optimizer, loss)
