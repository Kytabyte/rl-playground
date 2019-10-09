"""
An implementation of various replay buffer
"""
from typing import List

from .memory import Memory
from .utils import SumTree

import torch


class Replay(Memory):
    def __init__(self, fields=('state', 'action', 'next_state', 'reward', 'done'), cap: int = 1000):
        super(Replay, self).__init__(fields, cap)
        self._pos = 0

    def push(self, *args: torch.Tensor) -> None:
        if len(self) == self._cap:
            self[self._pos] = args
        else:
            self.append(args)
        self._pos = (self._pos + 1) % self._cap

    def push_batch(self, *args: torch.Tensor) -> None:
        bs = args[0].size(0)
        pos, size, cap = self._pos, len(self), self._cap
        if size == cap:
            if pos + bs <= cap:
                self[pos:pos + bs] = args
            else:
                self[pos:cap] = [field[:cap - pos] for field in args]
                self[:pos + bs - cap] = [field[cap - pos:] for field in args]
        else:
            if pos + bs <= cap:
                self.extend(args)
            else:
                self.extend([field[:cap - pos] for field in args])
                self[:pos + bs - cap] = [field[cap - pos:] for field in args]

        self._pos = (pos + bs) % cap


class PrioritizedReplay(Replay):
    def __init__(self, fields=('state', 'action', 'next_state', 'reward', 'done'), cap: int = 1000):
        super(PrioritizedReplay, self).__init__(fields, cap)
        self._weight = SumTree(cap)

    def get_priority(self, index):
        return self._weight.get(index)

    def set_priority(self, index, priority):
        self._weight.set(index, priority)

    def sample_index(self, batch_size: int, repeat: bool = False) -> List[int]:
        return self._weight.sample(batch_size)
