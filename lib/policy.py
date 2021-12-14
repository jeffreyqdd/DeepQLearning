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
    def __init__(self, start_epsilon:float=1.00, decay:float = 0.001) -> None:
        """ The constructor for the EpsilonGreedyPolicy class.

        ### Params
        1. start_epsilon (float) - starting probability of exploration
        2. decay (float) - linear unit to decrease epsilon by during every selection
        """
        self.epsilon = start_epsilon
        self.decay   = decay

        ### reuse old code (good coding habits) (Daisy Fan would be proud)
        self.greedy = GreedyPolicy()
        self.random = RandomPolicy()
    
    def select(self, action_weights: np.ndarray) -> int:
        choice = np.random.rand(1)[0]
        if choice < self.epsilon:
            # explore
            return self.random(action_weights)
        else:
            # exploit
            return self.greedy(action_weights)


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
        """
        self.temperature = temperature

if __name__ == '__main__':
    x = GreedyPolicy()
    x.select()
