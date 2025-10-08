REGISTRY = {}

def task(name=None):
    def wrap(fn):
        REGISTRY[name or fn.__name__] = fn
        return fn
    return wrap

from . import mpc