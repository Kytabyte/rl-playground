import random

import torch
import torch.nn as nn

from commons.torch_utils import astensor, copynet

class QNet():
    def __init__(self, net, n_obs, n_act, target=False):
        self._net = net
        self._tnet = copynet(net) if target else None
        
        self._n_obs, self._n_act = n_obs, n_act

    def parameters(self):
    	return self._net.parameters()
    
    def forward(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = astensor(obs, 'float')
        
        return self._net(obs)
    
    def learn(self, samples, optim, loss_fn, gamma=0.95):
        assert len(samples) == 5
        
        obs, act, next_obs, reward, done = samples

        if not isinstance(obs, torch.Tensor):
        	obs = astensor(obs, 'float')

        if not isinstance(act, torch.Tensor):
        	act = astensor(act, 'long')

        if not isinstance(next_obs, torch.Tensor):
        	next_obs = astensor(next_obs, 'float')

        if not isinstance(reward, torch.Tensor):
        	reward = astensor(reward, 'float')

        if not isinstance(done, torch.Tensor):
        	done = astensor(done, 'float')
        
        qval = self._net(obs)
        
        if self._tnet:
            opt_qval = self._tnet(next_obs).detach()
        else:
            opt_qval = self._net(next_obs).detach()
        
        qval = qval.gather(1, act.unsqueeze(1)).squeeze(1)
        opt_qval = opt_qval.max(1)[0]
        exp_qval = reward + gamma * opt_qval * (1 - done)
        
        loss = loss_fn(qval, exp_qval)
        optim.zero_grad()
        loss.backward()
        optim.step()
    
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
    
    def update_tnet(self):
        if self._tnet:
            self._tnet.load_state_dict(self._net.state_dict())
        else:
            print('You did not initialize target network, please check your network structure.')