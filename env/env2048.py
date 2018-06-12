
# coding: utf-8

# In[ ]:


from gym import error, spaces


# In[ ]:


class Env2048(object):
    def __init__(self):
        self._action_set = {0,1,2,3}
        self.action_space = spaces.Discrete(len(self._action_set))
    
    def step(self, action):
        pass
    
    def reset(self):
        pass
    
ACTION_MEANING = {
    0: 'UP',
    1: 'RIGHT',
    2: 'DOWN',
    3: 'LEFT'
}

