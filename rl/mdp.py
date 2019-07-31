import itertools
from typing import Optional, Tuple

import numpy as np

from envs import ObservableEnv


class _BaseMDP:
    def __init__(self,
                 env: ObservableEnv,
                 init_value: np.ndarray,
                 init_policy: Optional[np.ndarray],
                 gamma: float):
        self._env = env
        self._values = init_value
        self._pi = init_policy
        self._old_values = None
        self._old_pi = None
        self._gamma = gamma

    @property
    def value(self) -> np.ndarray:
        return self._values

    @property
    def policy(self) -> Optional[np.ndarray]:
        return self._pi

    def step(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError

    def converge(self) -> bool:
        raise NotImplementedError

    def run(self, n_iter: Optional[int] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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
                 init_value: np.ndarray,
                 init_policy: Optional[np.ndarray] = None,
                 gamma: float = 0.9):
        super(ValueIteration, self).__init__(env, init_value, init_policy, gamma)

    def step(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        self._old_values = self._values

        next_values = self._env.rewards + self._gamma * np.sum(self._env.transitions * self._old_values, axis=2)
        self._values, self._pi = np.max(next_values, axis=1), np.argmax(next_values, axis=1)
        return self._values, self._pi

    def converge(self) -> bool:
        if self._old_values is None:
            return False
        return np.allclose(self._old_values, self._values)


class PolicyIteration(_BaseMDP):
    def __init__(self,
                 env: ObservableEnv,
                 init_value: np.ndarray,
                 init_policy: Optional[np.ndarray],
                 gamma: float = 0.9):
        super(PolicyIteration, self).__init__(env, init_value, init_policy, gamma)

    def step(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        self._old_values, self._old_pi = self._values, self._pi

        self._values = np.zeros(self._old_values.shape)
        for state, policy in enumerate(self._old_pi):
            self._values[state] = self._env.rewards[state, policy] + self._gamma * np.sum(
                self._env.transitions[state, policy, :] * self._old_values)
        self._pi = np.argmax(self._env.rewards + self._gamma * np.sum(self._env.transitions * self._values, axis=2), axis=1)
        return self._values, self._pi

    def converge(self) -> bool:
        if self._old_pi is None:
            return False
        return np.all(self._old_pi == self._pi)


class ModifiedPolicyIteration(_BaseMDP):
    def __init__(self,
                 env: ObservableEnv,
                 init_value: np.ndarray,
                 init_policy: Optional[np.ndarray],
                 gamma: float = 0.9,
                 n_evals: int = 5):
        super(ModifiedPolicyIteration, self).__init__(env, init_value, init_policy, gamma)
        self._n_evals = n_evals

    def step(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        self._old_values, self._old_pi = self._values, self._pi

        self._values = np.zeros(self._old_values.shape)
        for _ in range(self._n_evals):
            for state, policy in enumerate(self._old_pi):
                self._values[state] = self._env.rewards[state, policy] + self._gamma * np.sum(
                    self._env.transitions[state, policy, :] * self._old_values)
            self._old_values = self._values
        next_values = self._env.rewards + self._gamma * np.sum(self._env.transitions * self._values, axis=2)
        self._values, self._pi = np.max(next_values, axis=1), np.argmax(next_values, axis=1)
        return self._values, self._pi

    def converge(self) -> bool:
        if self._old_values is None:
            return False
        return np.allclose(self._old_values, self._values)
