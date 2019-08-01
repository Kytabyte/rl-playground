from typing import List

import torch


class ObservableEnv:
    """
    """
    def __init__(self,
                 states: List[str],
                 actions: List[str],
                 transitions: torch.Tensor,
                 rewards: torch.Tensor):
        """
        """
        self._check_valid_input(states, actions, transitions, rewards)

        self._states = states
        self._actions = actions
        self._transitions = transitions
        self._rewards = rewards

    @staticmethod
    def _check_valid_input(states, actions, transitions, rewards):
        n_states, n_actions = len(states), len(actions)
        assert transitions.size() == (n_states, n_actions, n_states)
        assert rewards.size() == (n_states, n_actions)

    @property
    def states(self):
        return self._states

    @property
    def actions(self):
        return self._actions

    @property
    def transitions(self):
        return self._transitions

    @property
    def rewards(self):
        return self._rewards
