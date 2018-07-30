import random
import numpy as np
import torch 

class ReplayBuffer():
  def __init__(self, size):
    self._buffer = []
    self._size = size
    self._next_idx = 0

  def __len__(self):
    return len(self._buffer)

  def push(self, sample):
    assert len(sample) == 5

    if len(self._buffer) < self._size:
      self._buffer.append(sample)
    else:
      self._buffer[self._next_idx] = sample

    self._next_idx = (self._next_idx + 1) % self._size 

  def sample(self, n_sample):
    samples = random.sample(self._buffer, n_sample)
    samples = list(map(np.array, zip(*samples)))
    return samples
