"""

"""


class DotDict(dict):
    __getattr__ = dict.__getitem__
