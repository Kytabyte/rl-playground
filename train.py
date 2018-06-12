
# coding: utf-8

# In[1]:


from envs.env2048 import Env2048


# In[2]:


env = Env2048()


# In[3]:


for i_episode in range(20):
    observation = env.reset()
    for t in range(200):
        if t % 50 == 0:
            print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print(observation)
            print("Episode finished after {} timesteps".format(t+1))
            break

