from abc import ABC, abstractmethod 
import numpy as np

class Policy(ABC):
    """Policy class used to balance between exploration and exploitation.

    ### Note that these policies may fail if we get dynamic board states e.g tic tac toe
    """
    
    @abstractmethod
    def select(self, action_weights:np.ndarray) -> int:
        """Selects an action based on the weights. Policy varies by class.
        
        ### Params
        1. action_weights (np.ndarray) - weights for each possible action

        ### Returns
        1. (int) - selected action
        """
        raise NotImplementedError()


class GreedyPolicy(Policy):
    """Policy class that choses the action with the highest weight.
    """
    def select(self, action_weights: np.ndarray) -> int:
        return np.argmax(action_weights)



class RandomPolicy(Policy):
    """Policy class that randomly chooses an action.
    """
    def select(self, action_weights: np.ndarray) -> int:
        return np.random.choice(action_weights.size, 1, replace=True)[0]



class EpsilonGreedyPolicy(Policy):
    """Policy class that implements linear decay from exploration to exploitation.
    """
    def __init__(self, start_epsilon:float=1.00, end_epsilon:float=0.00, decay:float = 0.001) -> None:
        """ The constructor for the EpsilonGreedyPolicy class.

        ### Params
        1. start_epsilon (float) - starting probability of exploration
        2. end_epsilon (float) - minimum probability of exploration
        3. decay (float) - linear unit to decrease epsilon by during every selection
        """
        self.epsilon = start_epsilon
        self.min_ep  = end_epsilon
        self.decay   = decay

        ### reuse old code (good coding habits) (Daisy Fan would be proud)
        self.greedy = GreedyPolicy()
        self.random = RandomPolicy()
    
    def select(self, action_weights: np.ndarray) -> int:
        choice = np.random.rand(1)[0]
        if choice < self.epsilon:
            # explore
            action = self.random.select(action_weights)
        else:
            # exploit
            action = self.greedy.select(action_weights)
        
        # decay
        self.epsilon = max(self.epsilon - self.decay, self.min_ep)
        
        # return
        return action


class BoltzmannQPolicy(Policy):
    """Policy class that chooses an action based off the weight of the action.

    ### Note
    1. There is an inherent "exploration" factor especially at the beginning 
    where everything is uniform
    2. May not be the optimal exploration Policy for an end-of-training-life agent
    """

    def __init__(self, temperature:float=1.0):
        """ The constructor for the EpsilonGreedyPolicy class.

        ### Params
        1. start_epsilon (float) - starting probability of exploration
        2. decay (float) - linear unit to decrease epsilon by during every selection

        ### Note
        1. Lower temperature exacerbates differences between weights
        (new_weights = old_weights / temperature)
        2. keep temperature between (0, inf?) <-- inf not good idea though
        """
        self.temperature = temperature

    def select(self, action_weights: np.ndarray) -> int:
        # scale by temperature
        # vectorized code!
        new_weights = action_weights / self.temperature

        # softmax
        new_weights = self.__softmax(new_weights)

        # select
        return np.random.choice(
            a=new_weights.size,
            size=1,
            replace=True,
            p=new_weights
        )[0]

    def __softmax(self, arr: np.ndarray) -> np.ndarray:
        """Implementation of a softmax.

        ### Params
        1. arr (np.ndarray) - array of weights

        ### Returns
        1. (np.ndarray) - normalized array bwtn (0,1)
        """

        return np.exp(arr) / np.sum(np.exp(arr))
