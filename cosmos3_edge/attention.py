"""Cosmos joint GQA using the shared tiled native VM attention operator."""
from native_pi0.attention import ResidentAttention


class NativeJointAttention(ResidentAttention):
    """Noncausal generator queries over prepared understanding+generation KV.

    Q [rows,16,128], K/V [total_rows,8,128]. Caller prepares the cache's
    understanding prefix once; forward reads the combined cache without
    concatenation or recomputation. All tensor views below are setup-only.
    """
    def __init__(self, library, q, k, v, *, workers=148, splits=1, output=None):
        if q.ndim != 3 or tuple(q.shape[1:]) != (16, 128):
            raise ValueError('Cosmos Q must have shape [rows,16,128]')
        if k.ndim != 3 or tuple(k.shape[1:]) != (8, 128) or v.shape != k.shape:
            raise ValueError('Cosmos K/V must have shape [total_rows,8,128]')
        super().__init__(library, q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                         workers=workers, splits=splits, output=None if output is None else output.unsqueeze(0))
        self.batched_output = self.output
        self.output = self.output[0]
