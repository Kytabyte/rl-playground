"""
An implementation of memory information needed during agent learning
"""

import collections
import random
from typing import Union, Dict, Iterator, Iterable, List, Tuple, Optional

import torch


class _MemResDict(collections.OrderedDict):
    __getattr__ = collections.OrderedDict.__getitem__

    def __iter__(self):
        for key in super().__iter__():
            yield getattr(self, key)

    def __str__(self):
        # TODO: str and repr
        return super(_MemResDict, self).__str__()


class Memory:
    r"""
    Args:
        fields:
        cap:
    """
    __slots__ = ['_fields', '_cap', '_memory']

    def __init__(self,
                 fields: Union[Iterable[str]],
                 cap: int = 10):
        if not fields:
            raise ValueError('The length of `fields` must be positive, but got 0.')
        self._fields = tuple(fields)  # make fields unchangeable
        self._cap = cap
        self._memory = collections.OrderedDict({field: [] for field in fields})

    def _check_len(self, value):
        if len(value) != len(self._memory):
            raise ValueError('Expected %d fields but only got %d' % (len(self._memory), len(value)))

    def _convert_dict_to_list(self, dct):
        return [dct[field] for field in self._memory]

    def __str__(self) -> str:
        return '{}(fields={}, cap={}, size={})'.format(
            type(self).__name__,
            self.fields,
            self.cap,
            len(self)
        )

    def __repr__(self):
        return str(self)

    def __getitem__(self,
                    item: Union[int, slice, Iterable[int]]) -> _MemResDict:
        """

        Args:
            item:

        Returns:

        """
        if isinstance(item, int):
            return _MemResDict({
                field: mem[item] for field, mem in self._memory.items()
            })

        if isinstance(item, slice):
            return _MemResDict({
                field: torch.stack(mem[item]) for field, mem in self._memory.items()
            })

        if isinstance(item, (tuple, list)):
            return _MemResDict({
                field: torch.stack([mem[idx] for idx in item]) for field, mem in self._memory.items()
            })

        raise TypeError("%s indices must be int, slice, list, or tuple, but not %s"
                        % (type(self).__name__, type(item).__name__))

    def __setitem__(self,
                    item: Union[int, slice, Iterable[int]],
                    value: Union[Iterable[torch.Tensor], Dict[str, torch.Tensor]]) -> None:
        """

        Args:
            item:
            value:

        Returns:

        """
        if isinstance(value, dict):
            value = self._convert_dict_to_list(value)

        if not isinstance(value, (list, tuple)):
            raise TypeError('Assigned type must be list, tuple, or dict, not %s'
                            % type(value).__name__)

        self._check_len(value)

        if isinstance(item, (int, slice)):
            for field, val in zip(self._memory, value):
                self._memory[field][item] = val
        elif isinstance(item, (list, tuple)):
            for field, vals in zip(self._memory, value):
                for i, val in zip(item, vals):
                    self._memory[field][i] = val
        else:
            raise TypeError("%s indices must be int, slice, list, or tuple, but not %s"
                            % (type(self).__name__, type(item).__name__))

    def __delitem__(self, key: Union[int, slice, Iterable[int]]) -> None:
        """

        Args:
            key:

        Returns:

        """
        if isinstance(key, (int, slice)):
            for mem in self._memory.values():
                del mem[key]
        elif isinstance(key, (tuple, list)):
            for idx in key:
                for mem in self._memory.values():
                    del mem[idx]
        else:
            raise TypeError("%s indices must be int, slice, list, or tuple, but not %s"
                            % (type(self).__name__, type(key).__name__))

    def __len__(self):
        return len(self._memory[self._fields[0]])

    @property
    def cap(self) -> int:
        """

        Returns:

        """
        return self._cap

    @property
    def fields(self) -> Tuple[str]:
        """

        Returns:

        """
        return self._fields

    def pop(self, index: int = -1) -> _MemResDict:
        """

        Args:
            index:

        Returns:

        """
        result = self[index]
        del self[index]
        return result

    def append(self,
               item: Union[Iterable[torch.Tensor], Dict[str, torch.Tensor]]) -> None:
        """

        Args:
            item:

        Returns:

        """
        if len(self) >= self._cap:
            raise Exception('Memory size reaches capacity.')

        if isinstance(item, dict):
            item = self._convert_dict_to_list(item)

        if not isinstance(item, (list, tuple)):
            raise TypeError('input `item` type must be list or dict, not %s' % type(item).__name__)

        for field, value in zip(self._memory, item):
            self._memory[field].append(value)

    def extend(self,
               items: Union[Iterable[torch.Tensor], Dict[str, torch.Tensor]]) -> None:
        """

        Args:
            items:

        Returns:

        """
        if len(self) >= self._cap:
            raise Exception('Memory size reaches capacity.')

        if isinstance(items, dict):
            items = self._convert_dict_to_list(items)

        if not isinstance(items, (list, tuple)):
            raise TypeError('`items` type must be list or dict, not %s' % type(items).__name__)

        for field, item in zip(self._memory, items):
            self._memory[field].extend([value for value in item])

    def sample_index(self, batch_size: int, repeat: bool = False) -> List[int]:
        """

        Args:
            batch_size:
            repeat:

        Returns:

        """
        seq = [i for i in range(len(self))]
        if repeat:
            return [random.choice(seq) for _ in range(batch_size)]
        return random.sample(seq, k=batch_size)

    def sample(self,
               batch_size: int,
               index: bool = False,
               repeat: bool = False) -> Union[Tuple[List[int], _MemResDict], _MemResDict]:
        """

        Args:
            batch_size:
            index:
            repeat:

        Returns:

        """
        indices = self.sample_index(batch_size, repeat)
        if index:
            return indices, self[indices]
        return self[indices]

    def batches(self, batch_size: int, shuffle: bool = False) -> Iterator[_MemResDict]:
        """

        Args:
            batch_size:
            shuffle:

        Returns:

        """
        seq = [i for i in range(len(self))]
        if shuffle:
            random.shuffle(seq)
        for group in range(len(self) // batch_size):
            start = group * batch_size
            end = start + batch_size
            yield self[seq[start:end]]

    def all(self, shuffle: bool = False) -> _MemResDict:
        """

        Args:
            shuffle:

        Returns:

        """
        if shuffle:
            seq = [i for i in range(len(self))]
            random.shuffle(seq)
            return self[seq]
        return self[:]

    def reset(self) -> None:
        """

        Returns:

        """
        del self[:]

    def flush(self) -> _MemResDict:
        mem = self.all()
        self.reset()
        return mem


if __name__ == '__main__':
    memory = Memory(['a', 'b', 'c'])
