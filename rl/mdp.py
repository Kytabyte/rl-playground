import itertools
from typing import Optional, Tuple

import torch

from envs import ObservableEnv


class _BaseMDP:
    def __init__(self,
                 env: ObservableEnv,
                 init_value: torch.Tensor,
                 init_policy: Optional[torch.Tensor],
                 gamma: float):
        self._env = env
        self._values = init_value
        self._pi = init_policy
        self._old_values = None
        self._old_pi = None
        self._gamma = gamma

    @property
    def value(self) -> torch.Tensor:
        return self._values

    @property
    def policy(self) -> Optional[torch.Tensor]:
        return self._pi

    def step(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError

    def converge(self) -> bool:
        raise NotImplementedError

    def run(self, n_iter: Optional[int] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        for i in itertools.count(1):
            value, policy = self.step()
            converged = self.converge()
            if n_iter is None and converged:
                print('Converged in {} iterations.'.format(i))
                return value, policy
            if n_iter is not None and i == n_iter:
                print('Run {} iterations and the value is {} converged'.format(n_iter, '' if converged else 'not'))
                return value, policy


class ValueIteration(_BaseMDP):
    def __init__(self,
                 env: ObservableEnv,
                 init_value: torch.Tensor,
                 init_policy: Optional[torch.Tensor] = None,
                 gamma: float = 0.9):
        super(ValueIteration, self).__init__(env, init_value, init_policy, gamma)

    def step(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._old_values = self._values
        rewards, transitions = self._env.rewards, self._env.transitions

        self._values, self._pi = torch.max(rewards + self._gamma * torch.sum(transitions * self._old_values, 2), 1)
        return self._values, self._pi

    def converge(self) -> bool:
        if self._old_values is None:
            return False
        return torch.allclose(self._old_values, self._values)


class PolicyIteration(_BaseMDP):
    def __init__(self,
                 env: ObservableEnv,
                 init_value: torch.Tensor,
                 init_policy: Optional[torch.Tensor],
                 gamma: float = 0.9):
        super(PolicyIteration, self).__init__(env, init_value, init_policy, gamma)

    def step(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._old_values, self._old_pi = self._values, self._pi
        rewards, transitions = self._env.rewards, self._env.transitions

        self._old_pi.unsqueeze_(1)
        self._values = rewards.gather(1, self._old_pi) + self._gamma * torch.sum(
            transitions * self._old_values, 2).gather(1, self._old_pi)
        self._old_pi.squeeze_()
        self._values.squeeze_()

        self._pi = torch.argmax(rewards + self._gamma * torch.sum(transitions * self._values, 2), 1)
        return self._values, self._pi

    def converge(self) -> bool:
        if self._old_pi is None:
            return False
        return bool(torch.all(self._old_pi == self._pi))


class ModifiedPolicyIteration(_BaseMDP):
    def __init__(self,
                 env: ObservableEnv,
                 init_value: torch.Tensor,
                 init_policy: Optional[torch.Tensor],
                 gamma: float = 0.9,
                 n_evals: int = 5):
        super(ModifiedPolicyIteration, self).__init__(env, init_value, init_policy, gamma)
        self._n_evals = n_evals

    def step(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._old_values, self._old_pi = self._values, self._pi
        rewards, transitions = self._env.rewards, self._env.transitions

        self._old_pi.unsqueeze_(1)
        for _ in range(self._n_evals):
            self._values = rewards.gather(1, self._old_pi) + self._gamma * torch.sum(
                transitions * self._values, 2).gather(1, self._old_pi)
            self._values.squeeze_()
        self._old_pi.squeeze_()
        self._values.squeeze_()

        self._values, self._pi = torch.max(rewards + self._gamma * torch.sum(transitions * self._values, 2), 1)
        return self._values, self._pi

    def converge(self) -> bool:
        if self._old_values is None:
            return False
        return torch.allclose(self._old_values, self._values)
