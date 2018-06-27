# rl-playground

 This repository will provide implementation and experiments of some state-of-art reinforcement learning algorithms to train agents to play games provided. Ideally, this repository will provide an interface for extension where new algorithms and new games can be easily added. 



## Structure

[`games`](https://github.com/Kytabyte/rl-playground/tree/master/games) contains the logic for playing a game

[`envs`](https://github.com/Kytabyte/rl-playground/tree/master/envs) contains the environment of to play a game, where an agent can take action into the environment and receive feedback from the environment, including observation, reward, etc.

[`models`](https://github.com/Kytabyte/rl-playground/tree/master/models) is a folder to place pure network structure implementation. For instance, pre-defined networks used in RL are placed in this folder. A model builder is also provided in this folder to generate some typical network structure. Further use of this folder can be found [here](https://github.com/Kytabyte/rl-playground/tree/master/models). Note that this folder is **not** for place RL algorithms. We detach the network structure from RL part as for providing more flexibility on building RL algorithms, and focusing on RL  itself when building algorithms.

[`rl`](https://github.com/Kytabyte/rl-playground/tree/master/rl) contains the RL algorithms for training an agent. The idea to implement an RL algorithm is treating the model (that take observation in and return action out) as an object inside, and use the method in this model to take action or learn model.



## Train

We need write a script like [this](https://github.com/Kytabyte/rl-playground/blob/master/train.py) to train an agent with a bunch of constant defined inside as follows.

1. Define an evrionment from `envs` folder, and get `n_obs` and `n_act` to initialize a network and a RL algorithm.
2. Initialize a network, which can be either a network in `networks` or a user-defined work, but should have `forward` and `learn` method.
3. Initialize an agent
4. Define all constants and train the agent



## Contribution

To create a new game, one can write a game logic and put into `games` and write an environment to interact with the game code, and put into `envs`. The only method required in the `env` class is `step`, which take an `action` and returns `observation`, `reward`, `done`, and `info` back.

