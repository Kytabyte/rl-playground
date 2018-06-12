import random

class Discrete(object):
    def __init__(self, size):
        self._size = size
        self._sample_space = [i for i in range(self._size)]
        
    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, self._size)
    
    def __repr__(self):
        return self.__str__()
    
    def sample(self):
        return random.choice(self._sample_space)
