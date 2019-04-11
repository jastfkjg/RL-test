import matplotlib.pyplot as plt 
import os.path as osp
import json
import os
import numpy as np 
import pandas


def smooth(y, radius, mode='two_sided', valid_only=False):
    '''
    smooth signal y, where radius is determines the size of the window

    mode='two_sided': average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal': average over the window [max(index - radius, 0), index]
    valid_only: put nan in entries where the full-sized window is not available
    '''
    assert mode in ('two_sided', 'causal')
    if len(y) < 2 * radius + 1:
        
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius + 1)
        out = np.convolve(y, convkernel, mode='same') / np.convolve(np.ones_like(y), convkernel,
                mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode=='causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel, mode='full') / np.convolve(np.ones_like(y), convkernel,
                mode='full')
        out = out[:-radius + 1]
        if valid_only:
            out[:radius] = np.nan
    return out


def plot_result(
        allresults, *,
        xy_fn=default_split_fn,
        )



