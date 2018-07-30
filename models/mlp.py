"""
  A collection of linear neural network
"""
import torch.nn as nn

"""
  class MLP is a interface to create any feed-forward
  neural network
"""
class MLP(nn.Module):
  def __init__(self, n_input, n_output, hidden=(128,)):
    super(MLP, self).__init__()
    self.layers = nn.Sequential(
        *self._buildnet(n_input, n_output, hidden)
    )

  """
    helper function to build network structure
  """
  def _buildnet(self, n_input, n_output, hidden):
    if not hidden:
      return [nn.Linear(n_input, n_output)]

    net = [nn.Linear(n_input, hidden[0]), nn.ReLU()]
    for n_in, n_out in zip(hidden[:-1], hidden[1:]):
      net.extend([nn.Linear(n_in, n_out), nn.ReLU()])
    net.append(nn.Linear(hidden[-1], n_output))
    return net

  def forward(self, x):
    return self.layers(x)
