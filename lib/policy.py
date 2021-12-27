from abc import ABC, abstractmethod 
import numpy as np

class Policy(ABC):
    """Policy class used to balance between exploration and exploitation.

    ### Note that these policies may fail if we get dynamic board states e.g tic tac toe
    """
    
    @abstractmethod
    def select(self, action_weights):
        """Selects an action based on the weights. Policy varies by class.
        
        ### Params
        1. action_weights (np.ndarray) - weights for each possible action

        ### Returns
        1. (int) - selected action
        """
        raise NotImplementedError()

    def get_config(self):
        """Returns class configuration"""
        return {}

class GreedyQPolicy(Policy):
    """Policy class that choses the action with the highest weight.
    """
    def select(self, action_weights):
        return np.argmax(action_weights)


class RandomQPolicy(Policy):
    """Policy class that randomly chooses an action.
    """
    def select(self, action_weights):
        return np.random.choice(action_weights.size, 1, replace=True)[0]



class EpsilonGreedyQPolicy(Policy):
    """Policy class that implements linear decay from exploration to exploitation.
    """
    def __init__(self, epsilon):
        """ The constructor for the EpsilonQGreedyPolicy class.

        ### Params
        1. epsilon (float) - probability of exploration
        """
        self.epsilon = epsilon

        ### reuse old code (good coding habits) (Daisy Fan would be proud)
        self.greedy = GreedyQPolicy()
        self.random = RandomQPolicy()
    
    def select(self, action_weights: np.ndarray) -> int:
        choice = np.random.rand(1)[0]
        if choice < self.epsilon:
            # explore
            action = self.random.select(action_weights)
        else:
            # exploit
            action = self.greedy.select(action_weights)
        
        # return
        return action

    def get_config(self):
        ret = super().get_config()
        ret['epsilon'] = self.epsilon
        return ret

class BoltzmannQPolicy(Policy):
    """Policy class that chooses an action based off the weight of the action.

    ### Note
    1. There is an inherent "exploration" factor especially at the beginning 
    where everything is uniform
    2. May not be the optimal exploration Policy for an end-of-training-life agent
    """

    def __init__(self, temperature=1.0, clip=(-500., 500.)):
        """ The constructor for the BoltzmannQPolicy class.

        ### Params
        1. temperature (float) - scaling factor for input weights
        2. declipcay (Tuple[int,int]) - min and max values after applying temperature

        ### Note
        1. Lower temperature exacerbates differences between weights
        (new_weights = old_weights / temperature)
        2. keep temperature between (0, inf?) <-- inf not good idea though
        """
        self.temperature = temperature
        self.clip = clip

    def select(self, action_weights):
        # scale by temperature
        # vectorized code!
        new_weights = action_weights / self.temperature
        new_weights = np.clip(new_weights, self.clip[0], self.clip[1])


        # softmax
        probabilities = self.__softmax(new_weights)

        # select
        return np.random.choice(
            a=probabilities.size,
            size=1,
            replace=True,
            p=probabilities
        )[0]

    def __softmax(self, arr: np.ndarray) -> np.ndarray:
        """Implementation of a softmax.

        ### Params
        1. arr (np.ndarray) - array of weights

        ### Returns
        1. (np.ndarray) - normalized array bwtn (0,1)
        """

        return np.exp(arr) / np.sum(np.exp(arr))
