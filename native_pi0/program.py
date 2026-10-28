"""Host compilation of static tile-range streams for the original Hazy VM."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearWork:
    tiles:int
    k:int


def linear_streams(stages,workers):
    """Gemma-style contiguous ranges assigned to least-loaded worker streams.

    Each instruction is [opcode, stage, first tile, tile count, ...]. Work is
    assigned once during setup; inference does not fetch from a global queue.
    """
    if not isinstance(workers,int) or not 1<=workers<=148:raise ValueError('Expected 1..148 B200 workers')
    stages=tuple(stages)
    if not stages or any(s.tiles<1 or s.k<64 or s.k%64 for s in stages):raise ValueError('Invalid linear stage geometry')
    streams=[[] for _ in range(workers)];cost=[0]*workers
    for stage,work in enumerate(stages):
        active=min(workers,work.tiles);base,remainder=divmod(work.tiles,active)
        available=sorted(range(workers),key=lambda worker:(cost[worker],worker));first=0
        for chunk,worker in enumerate(available[:active]):
            count=base+(chunk<remainder);code=[0]*32;code[:4]=[1,stage,first,count]
            streams[worker].append(code);cost[worker]+=count*work.k;first+=count
        assert first==work.tiles
    length=max(map(len,streams));result=np.zeros((workers,length,32),dtype=np.int32)
    for worker,stream in enumerate(streams):
        if stream:result[worker,:len(stream)]=stream
    return result


def program_summary(code):
    live=code[:,:,0]!=0
    return {'workers':int(code.shape[0]),'stream_length':int(code.shape[1]),'instructions':int(live.sum()),
        'compute_tiles':int(code[:,:,3][live].sum()),'largest_tile_range':int(code[:,:,3].max()),
        'dynamic_fetch':False,'ready_queue':False}
