from models.mlp import MLP
from envs.env2048 import Env2048

env = Env2048()
n_obs, n_act = env.n_obs(), len(env.action_space)
n_episode = 20

model = MLP(n_input=n_obs, n_output=n_act, endswith=nn.Softmax(dim=0))
net = PolicyNet(net=model,n_obs=n_obs, n_act=n_act)
optimizer = optim.Adam(net.parameters(), lr=1e-2)


from itertools import count


for i_eps in count(1):
    obs = env.reset()
    valid_move = None
    done = False
    
    for step in count(1):
        action = net.act(obs.flatten(), act_mask=valid_move)
        obs, reward, done, info = env.step(action)
        net.rewards.append(reward * (1 - int(done)))
        valid_move = info.get('valid_move', None)
        
        if step % 10000 == 0:
            print(obs,action, valid_move, done)
            
        if done:
            break
            
    print(obs)
    print('episode {} finished with step {} and score {}'.format(i_eps, step, reward))
    
    net.learn(optimizer, 0.99)
    obs = env.reset()
    valid_move = None
    done = False
    
    if i_eps > n_episode:
        break
