"""Native π0 operators and the complete BF16 reference model."""

def __getattr__(name):
    if name=='NativePi0':
        from native_pi0.model import NativePi0
        return NativePi0
    raise AttributeError(name)
