import torch
import torch.nn as nn
import torch.optim as optim

from commons.torch_utils import astensor
import commons.constants as const


class PolicyNet():
    def __init__(self, net, n_obs, n_act):
        self._net = net
        self._n_obs, self._n_act = n_obs, n_act
        
        self.logprobs = []
        self.rewards = []
    
    def forward(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = astensor(obs, 'float')
        
        return self._net(obs)
    
    def parameters(self):
        return self._net.parameters()

    def learn(self, optimizer, gamma):
        assert len(self.logprobs) > 0 and len(self.rewards) > 0
        assert len(self.logprobs) == len(self.rewards)
        
        mean, std = astensor(self.rewards,'float').mean().item(), astensor(self.rewards,'float').std().item()
        
        gain = 0
        loss = []
        while len(self.rewards) > 0:
            logprob, reward = self.logprobs.pop(), (self.rewards.pop() - mean) / (std + const.MACHINE_EPS)
            gain = reward + gamma * gain
            loss.append(-gain * logprob)
            
        optimizer.zero_grad()
        loss = torch.cat(loss).sum()
        loss.backward()
        optimizer.step()
        
    
    def act(self, obs, act_mask=None):
        act_mask = astensor([1] * self._n_act, 'float') if act_mask is None else astensor(act_mask, 'float')
        
        probs = self.forward(obs)
        print(probs)
        action = (probs * act_mask.unsqueeze(0)).argmax().item()
        
        self.logprobs.append(torch.log(probs))
        return action