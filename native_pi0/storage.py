"""Storage for packed inputs read through ThunderKittens swizzled TMA maps."""
import math

import torch


def packed_tma_buffer(shape, *, device):
    """Keep the final rounded 128-byte TMA sector inside initialized storage.

    The pinned TK map rounds its innermost dimension to the swizzle width,
    while retaining the logical row stride. A partial final sector can read
    up to 127 bytes beyond the logical last row. Its extra FP4 lanes have zero
    block scales; backing those bytes is still required for a legal load.
    """
    count=math.prod(shape)
    storage=torch.zeros(count+127,dtype=torch.uint8,device=device)
    return storage[:count].view(shape)
