from collections import namedtuple
import numpy as np
import warnings
import random
import copy

from numpy.lib.arraysetops import isin

# Transition state data storage.
# - state0,     starting state
# - action,     action perfromed at state0
# - reward,     reward received at state1
# - state1,     ending state
# - terminal1,  state1 might be terminal

Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')


############################################################
###### Utility Functions
############################################################
def sample_batch_idx(low, high, size):
    """Uniform random generation of integers in a range [low, high). 
    ### Params:
        low (int) - minimum value 
        high (int) - maximum value
        size (int) - number of integers to pick
    ### Returns:
        (np.ndarray) - list of length size ofrandomly generated integers between low and high

    Note: Returns np.ndarray when number of samples exceeds sampling range. May lead to
    undesired results.
    """
    # assert precondition that high > low
    assert high > low

    # choose sampling method
    if high - low >= size:
        # sample without replacement
        # https://github.com/numpy/numpy/issues/2764 is fixed
        idxs    = np.random.choice(high-low, size=size, replace=False)
        idxs    += low
    else:
        # sample with replacement 
        warnings.warn(f'Sampling {size} times from range [{low}, {high}). Sampling without replacement.')
        idxs    = np.random.randint(low, high-1, size=size)
    return idxs


############################################################
###### Replay Buffer Utility Class
############################################################

class CircularBuffer:
    """Array of fixed length and type. Oldest entry beyond the fixed size
    is replaced by the newest entry.
        - O(1) append
        - O(1) access

    Note: indexing out of bounds will not crash since modulo is employed
    for internal head-pointer wrapping. It may lead to undesired results. 
    Please check for out of bounds before accessing the data.

    Note: Uses python [] arrays instead of numpy arrays. 
        - slower init time (negligible)
        - lower memory footprint
        - faster append
        - faster access
    """
    def __init__(self, max_size):
        """Constructor.
        ### Params:
            max_size (int) - holds the max_size most recent elements
        """
        self.max_size   = max_size
        self.buffer     = [None] * self.max_size

        self.head_pntr  = 0
        self.size       = 0

    def append(self, item):
        """Append to the end of CircularBuffer.
        ### Params:
            item (any) - item to append
        """
        # append
        self.buffer[self.head_pntr] = item

        # update head
        self.head_pntr              += 1
        self.head_pntr              %= self.max_size
        
        # increase size
        self.size                   = min(self.size+1, self.max_size)

    def __getitem__(self, key):
        """Returns indexed items.
        ### Params:
            key (int or np.ndarray of ints) - index or indices
        ### Returns:
            (any) - singular item or list of items corresponding to the index/indices
        """
        # note that deque[0] is the oldest entry and deque[end] is the newest entry
        # need to slide indices because CircularBuffers employs a wrapping pointer

        # note that head_pntr points to the oldest entry and head_pntr + 1
        # points to the newest entry

        # can use vectorized code (yay numpy)
        key = (key + self.head_pntr + 1) % self.size

        sample_size = len(key)
        ret = [None] * sample_size

        for idx in range(sample_size):
            ret[idx] = self.buffer[key[idx]]

        return ret

