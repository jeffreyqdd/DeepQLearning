from typing import Tuple
import numpy as np
import warnings

class Memory:
    """An implementation of the replay buffer. 

    The class is designed to be as fast as possible.
    1. employs fixed-size numpy arrays. 
    2. queue-like implementation
    3. O(1) insertion/deletion
    4. O(1) random selection 
    """

    def __init__(self, input_dims: Tuple[int, ...], output_dims:Tuple[int, ...], max_size:int = 10_000) -> None:
        """ The constructor for the Memory class.

        ### Params
        1. input_dims (Tuple[int, ...]) - observation space size
        2. input_dims (Tuple[int, ...]) - action_space size
        3. max_size (int) - maximum replay buffer size

        ### Note
        let e_t = (s_t, a_t, r_t+1, s_t+1)
        - e_t is the agent's experience (transition state at time t)
        - s_t is the state at time t
        - a_t is the action taken at time t
        - r_t+1 is the reward received after performing the action
        - s_t+1 is the new state after performing the action
        - I will include a terminal array to keep track of terminal states
        - I will include a weights array to keep track of action weights at each state s_t
        """

        self.max_size = max_size
        self.mem_pntr = 0
        self.size = 0

        self.state          = np.zeros( shape = (self.max_size, *input_dims) , dtype = np.float32)
        self.action         = np.zeros( shape = (self.max_size)              , dtype = np.int32  )
        self.reward         = np.zeros( shape = (self.max_size)              , dtype = np.float32)
        self.new_state      = np.zeros( shape = (self.max_size, *input_dims) , dtype = np.float32)
        self.terminal       = np.zeros( shape = (self.max_size)              , dtype = bool      ) 
        self.action_weights = np.zeros( shape = (self.max_size, *output_dims), dtype = np.float32)
    
    def add_transition(self, state:np.ndarray, action:np.int32, reward:np.float32, new_state:np.ndarray,
                        is_done:bool, action_weights:np.ndarray) -> None:
        """Add new transition state to memory.
        ### Params
        1. state (np.ndarray) - current state
        2. action (np.int32) - action
        3. reward (np.float32) - reward value of action
        4. new_state (np.ndarray) - new state, may be empty
        5. is_done (bool) - true if new_state is a terminal state
        6. action_weights (np.ndarray) - weights for each action
        """
        self.state[self.mem_pntr]           = state
        self.action[self.mem_pntr]          = action
        self.reward[self.mem_pntr]          = reward
        self.new_state[self.mem_pntr]       = new_state
        self.terminal[self.mem_pntr]        = is_done
        self.action_weights[self.mem_pntr]  = action_weights

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
        1. (np.ndarray) - array of length sample_size where item i is (state  s_t)
        2. (np.int32)   - array of length sample_size where item i is (action a_t)
        3. (np.float32) - array of length sample_size where item i is (reward r_t+1)
        4. (np.ndarray) - array of length sample_size where item i is (state  s_t+1)
        5. (np.ndarray) - array of length sample_size where item i is (isdone s_t+1)
        6. (np.ndarray) - array of length sample_size where item i is (action_weights for a_t)
        """
        #TODO: Note that this ReplayMemory techically stores every thing sequentially. Could implement future fucntions to map transition_state t+1 to curr transition state

        if sample_size > self.size:
            warnings.warn(
                f"Sample size of {sample_size} is larger than the current memory size." +
                f" " +
                f"Returning empty sample."
            )

            none = np.array([])
            
            return none, none, none, none, none, none
        else:
            # sampling with replacement for performance gains
            batch = np.random.choice(self.size, sample_size, replace=True)

            states = self.state[batch]
            actions = self.action[batch]
            rewards = self.reward[batch]
            new_states = self.new_state[batch]
            terminals = self.terminal[batch]
            action_weights = self.action_weights[batch]

            return states, actions, rewards, new_states, terminals, action_weights

def main():
    from lib.moduletest import BetterUnittest
    import gym

    class TestMemory(BetterUnittest):
        def test_stability(self):
            """This test method checks the stability of my memory class. It should catch all potential crashes."""

            ### create env
            env = gym.make('CartPole-v1')
            observation = env.reset()

            OBS_SHAPE = env.observation_space.shape
            ### create memory
            replay_memory = Memory(input_dims=OBS_SHAPE, max_size=1_000)
            for _ in range(20_000):
                action = env.action_space.sample()
                new_observation, reward, done, info = env.step(action)
                
                if done:
                    new_observation = env.reset()

                replay_memory.add_transition(observation, action, reward, new_observation, done, np.array([0.1, 0.23]))
                observation = new_observation

                if replay_memory.size > 900:
                    x = replay_memory.sample_transitions(sample_size=64)
                    self.assertEqual(len(x) == 6)

            env.close()

    TestMemory.betterRun("MEMORY.PY @ MAIN ..running unittests")

if __name__ == '__main__':
    main()