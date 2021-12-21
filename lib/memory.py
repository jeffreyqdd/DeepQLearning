from typing import Tuple
import unittest
import numpy as np
import warnings

class Memory:
    """An implementation of the replay buffer. 

    ### The class is designed to be as fast as possible.
    1. employs fixed-size numpy arrays. 
    2. rolling-pointer implementation --> oldest entry is replaced
    3. O(1) insertion/deletion
    4. O(1) random selection 

    ### Note: if current state is at index i, then future state is at index i + 1 
    - True if and only if current state is not the oldest entry

    ### Note: A transition state implementation looks like this: 
    - e_t is the agent's experience (transition state at time t)
    - s_t is the state at time t
    - a_t is the action taken at time t
    - r_t+1 is the reward received after performing the action
    - s_t+1 is the new state after performing the action    
    - I will include a terminal array to keep track of terminal states
    - I will include a weights array to keep track of action weights at each state s_t
        and s_t+1
    """
     
    def __init__(self,
        input_dims: Tuple[int, ...],
        output_dims:Tuple[int, ...],
        max_len:int = 10_000
    ) -> None:
        """ The constructor for the Memory class.

        ### Params
        1. input_dims (Tuple[int, ...]) - observation space size
        2. input_dims (Tuple[int, ...]) - action_space size
        3. max_len (int) - maximum replay buffer size
        """

        self.max_size = max_len
        self.mem_pntr = 0
        self.size = 0

        self.curr_state     = np.zeros( shape = (self.max_size, *input_dims) , dtype = np.float32)
        self.actions        = np.zeros( shape = (self.max_size)              , dtype = np.int32  )
        self.rewards        = np.zeros( shape = (self.max_size)              , dtype = np.float32)
        self.future_state   = np.zeros( shape = (self.max_size, *input_dims) , dtype = np.float32)
        self.terminal       = np.zeros( shape = (self.max_size)              , dtype = bool      ) 
        self.curr_qs        = np.zeros( shape = (self.max_size, *output_dims), dtype = np.float32)

    
    def add_transition(self,
        curr_state:np.ndarray,
        action:np.int32,
        reward:np.float32,
        future_state:np.ndarray,
        is_done:bool,
        curr_qs:np.ndarray,
    ) -> None:
        """Add new transition state to memory.
        ### Params
        1. curr_state (np.ndarray) - current state
        2. action (np.int32) - action
        3. reward (np.float32) - reward value of action
        4. future_state (np.ndarray) - new state, may be gibberish
        5. is_done (bool) - true if new_state is a terminal state
        6. curr_qs (np.ndarray) - q weights for the current state
        """
        self.curr_state[self.mem_pntr]      = curr_state
        self.actions[self.mem_pntr]         = action
        self.rewards[self.mem_pntr]         = reward
        self.future_state[self.mem_pntr]    = future_state
        self.terminal[self.mem_pntr]        = is_done
        self.curr_qs[self.mem_pntr]         = curr_qs

        # increment and wrap around
        self.mem_pntr += 1
        self.mem_pntr %= self.max_size
        
        # because we increment and wrap around, the size caps at max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_transitions(self, sample_size:int) -> np.ndarray:
        """Uniformly samples from the entire array.

        ### Params
        1. sample_size (int) - number of samples to take from the memory

        ### Returns (note that the dtypes listed below are wrapped in an np.ndarray (batch sample). )
        1. (np.ndarray) - array of length sample_size where item i is (current_state  s_t)
        2. (np.int32)   - array of length sample_size where item i is (action a_t)
        3. (np.float32) - array of length sample_size where item i is (reward r_t+1)
        4. (np.ndarray) - array of length sample_size where item i is (future_state  s_t+1)
        5. (np.ndarray) - array of length sample_size where item i is (isdone s_t+1)
        6. (np.ndarray) - array of length sample_size where item i is (current_qs for a_t)
        7. (np.ndarray) - array of length sample_size where item i is (future_qs for a_t+1)
        """
        BUFFER = 2

        if sample_size > self.size + BUFFER:
            warnings.warn(
                f"Sample size of {sample_size} is larger than the current memory size." +
                f" " +
                f"Returning empty sample."
            )

            none = np.array([])
            
            return none, none, none, none, none, none
        else:
            batch = np.random.choice(self.size - BUFFER, sample_size, replace=False)
            batch = (batch + self.mem_pntr + 1) % self.size # shift end to pointer

            return (
                self.curr_state[batch],
                self.actions[batch],
                self.rewards[batch],
                self.future_state[batch],
                self.terminal[batch],
                self.curr_qs[batch],
                self.curr_qs[(batch + 1) % self.size]
            )
if __name__ == '__main__':
    from lib.tools import PrintingUtils
    import unittest
    import gym
    from tqdm import tqdm
    
    class TestMemory(unittest.TestCase, PrintingUtils):
        def test_stability(self):
            """This test method checks the stability of my memory class. It should catch all potential crashes."""
            self.assertTrue(True)
            ### create env
            env = gym.make('CartPole-v1')
            observation = env.reset()

            OBS_SHAPE = env.observation_space.shape
            ACTION_SHAPE = (env.action_space.n, )
            ### create memory
            replay_memory = Memory(input_dims=OBS_SHAPE, output_dims=ACTION_SHAPE,max_len=1_000)
            for _ in tqdm(range(100_000)):
                action = env.action_space.sample()
                new_observation, reward, done, info = env.step(action)
                
                if done:
                    new_observation = env.reset()

                replay_memory.add_transition(observation, action, reward, new_observation, done, np.array([0.1, 0.23]))
                observation = new_observation

                if replay_memory.size >= 90:
                    x = replay_memory.sample_transitions(sample_size=64)

            env.close()

    TestMemory.pretty_print("MEMORY.PY @ MAIN ..running unittests")
    unittest.main()