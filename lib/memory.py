from collections import namedtuple
from abc import ABC, abstractclassmethod
import numpy as np
import warnings

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
        * low (int) - minimum value 
        * high (int) - maximum value
        * size (int) - number of integers to pick

    ### Returns:
        * (np.ndarray) - list of length size ofrandomly generated integers between low and high

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
        idxs    = np.random.randint(low, high, size=size)
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
            * max_size (int) - holds the max_size most recent elements
        """
        self.max_size   = max_size
        self.buffer     = [None] * self.max_size

        self.head_pntr  = 0
        self.size       = 0

    def append(self, item):
        """Append to the end of CircularBuffer.

        ### Params:
            * item (any) - item to append
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
            * key (int or np.ndarray of ints) - index or indices

        ### Returns:
            * (any) - singular item or list of items corresponding to the index/indices
        """
        # note that deque[0] is the oldest entry and deque[end] is the newest entry
        # need to slide indices because CircularBuffers employs a wrapping pointer

        # note that head_pntr points to the oldest entry and head_pntr + 1
        # points to the newest entry

        # can use vectorized code (yay numpy)
        key = self.__wrap(key)

        if not hasattr(key, '__len__'):
            sample_size = 1
            key = [key] # package in array
        else:
            sample_size = len(key)
        
        ret = [None] * sample_size

        for idx in range(sample_size):
            ret[idx] = self.buffer[key[idx]]
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def to_list(self):
        """Returns list format of internal buffer"""
        arr = [None] * self.size #init list

        for i in range(self.size):
            key = self.__wrap(i)
            arr[i] = self.buffer[key]
        return arr

    def __wrap(self, idx):
        """Shifts the index to the true location in the buffer"""
        
        if self.size == 0:
            raise ValueError("Cannot index empty buffer")

        # can use vectorized code
        return (idx + self.head_pntr) % self.size

    def __len__(self):
        """Returns length of the CircularBuffer.
        """
        return self.size

    def __str__(self):
        """ToString"""
        return str(self.to_list)

    def __repr__(self):
        """Returns string of self"""
        return self.__str__()

############################################################
###### Parent Memory Class
############################################################
class Memory(ABC):
    """Abstract replay memory class.
    """
    @abstractclassmethod
    def sample(self, batch_size):
        """Samples replay memory.

        ### Params:
            * batch_size (int) - sample size

        ### Returns:
            * (list) - transition states Experience(state, action, reward, new_state)
        """
        raise NotImplementedError()

    @abstractclassmethod
    def append(self, state, action, reward, new_state, terminal):
        """Add to replay memory.

        ### Params:
            * state (np.ndarray) - current state
            * action (int) - action taken at current state
            * reward (float) - reward received after taking action
            * new_state (np.ndarray) - state after taking action
            * terminal (bool) - is state terminal
        """
        raise NotImplementedError()

    def get_config(self):
        """Get class configuration.

        ### Returns:
            * (dict) - class configuration
        """
        return {}

############################################################
###### Uniform Replay Memory
############################################################
class UniformMemory(Memory):
    """Uniformly samples from replay memory.
    """
    def __init__(self, max_size):
        self.max_size = max_size
        self.memory   = CircularBuffer(max_size=self.max_size)

    def sample(self, batch_size):
        idxs = sample_batch_idx(0, len(self.memory), size=batch_size)
        return self.memory[idxs]

    def append(self, state, action, reward, new_state, terminal):
        self.memory.append(Experience(
            state0=state,
            action=action,
            reward=reward,
            state1=new_state,
            terminal1=terminal,
        ))

    def get_config(self):
        ret = super().get_config()
        ret['max_size'] = self.max_size
        return ret

############################################################
###### Prioritized Replay Memory (TODO)
############################################################

# https://towardsdatascience.com/how-to-implement-prioritized-experience-replay-for-a-deep-q-network-a710beecd77b
# https://arxiv.org/abs/1511.05952
class PrioritizedMemory(Memory):
    """Implement me"""
    Priority = namedtuple("Priority", "priority, probability, weight, index")
    pass  