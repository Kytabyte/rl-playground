"""
  utils
"""
import copy

import torch
import torch.nn as nn
from . import constants as const

def astensor(obj, to_type=None):
  """
    to_type selected from 'int', 'long', 'float', 'double', and 'byte'

    not responsible for handling type cast error
  """
  if to_type is None:
    return torch.tensor(obj).to(const.DEVICE) if not isinstance(obj, torch.Tensor) else obj
  elif to_type == 'int':
    return torch.IntTensor(obj).to(const.DEVICE) if not isinstance(obj, torch.Tensor) else obj.int()
  elif to_type == 'long':
    return torch.LongTensor(obj).to(const.DEVICE) if not isinstance(obj, torch.Tensor) else obj.long()
  elif to_type == 'float':
    return torch.FloatTensor(obj).to(const.DEVICE) if not isinstance(obj, torch.Tensor) else obj.float()
  elif to_type == 'double':
    return torch.DoubleTensor(obj).to(const.DEVICE) if not isinstance(obj, torch.Tensor) else obj.double()
  elif to_type == 'byte':
    return torch.ByteTensor(obj).to(const.DEVICE) if not isinstance(obj, torch.Tensor) else obj.byte()
  else:
    raise TypeError(
      '''
        to_type can only be one of 
        'int', 'long', 'float', 'double', 'byte'
        or leave empty.
      '''
      )

def copynet(network):
  """
    a deepcopy of a network 
  """
  if not isinstance(network, nn.Module):
    raise TypeError('Input must be a instance of {}, but got {}'.format(nn.Module, network.__class__.__name__))

  return copy.deepcopy(network)


def flatten(network, input_size):
  """
    return the number of features after applying `network`structure on a network with
    shape `input_size`
  """
  if not isinstance(network, nn.Module):
    network = nn.Sequential(
        *network
    )

  with torch.no_grad():
    num_features = network(torch.rand((1, *input_size))).view(1, -1).size(1)

  return num_features
