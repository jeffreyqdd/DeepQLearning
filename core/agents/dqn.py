from core.memory import Memory
from core.policy import BoltzmannQPolicy, EpsilonGreedyQPolicy
from core.agent import BaseAgent

from keras import models

import numpy as np

class DqnAgent(BaseAgent):
    """Implementation of a deep q-learning agent.
    """
    def __init__(self, model_factory, memory, policy, discount_rate, nb_warmup_steps, target_update_rate, train_every):

        """Constructor of the Agent class
        ### Params
        1. model  (function)  - function with no arguments that returns a keras Model
        2. memory (Memory) - replay memory object from the library
        3. policy (Policy) - policy object from the library
        4. learning_rate (float) - learning rate of the deep q agent
        5. discount_rate (float) - discount rate of the deep q agent
        6. update_rate (int) - update frequency of target model from learn calls
        """
        self.memory = memory     
        self.policy = policy
        self.model  = model_factory()
        self.target_model = model_factory()
        self.target_model.set_weights(self.model.get_weights())

        self.gamma = discount_rate

        self.nb_warmup_steps = nb_warmup_steps

        self.target_model_update = target_update_rate
        self.update_cntr = 0

        self.train_every = train_every

    def forward(self, observation):
        action = self.target_model.predict(np.expand_dims(observation, axis=0))[0]
        return self.policy.select(action_weights=action)
    

    def backward(self, state, action, reward, new_state, terminal):
        self.memory.append(state, action, reward, new_state, terminal)

        if self.step % self.train_every == 0:
            pass

    #     X = []
    #     Y = []

    #     for idx in range(batch_size):
    #         current_state = curr_states[idx]
    #         action        = actions[idx]
    #         reward        = rewards[idx]
    #         done          = terminals[idx]
            
    #         # (reward + dr * Q(s_t+1, a_t) )
    #         if not done:
    #             best_future_q = np.max(future_qs[idx])
    #             new_q = reward + self.dr * best_future_q
    #         else:
    #             new_q = reward
            
    #         adjust_qs = curr_qs[idx]
    #         adjust_qs[action] = new_q

    #         X.append(current_state)
    #         Y.append(adjust_qs)
    #     self.model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0)

    #     self.update_cntr += 1
    #     if self.update_cntr >= self.update_rate:
    #         self.update_cntr = 0
    #         self.target_model.set_weights(self.model.get_weights())

    # def save_model(self, path:str) -> None:
    #     """Save trained weights and model architecture to file.

    #     ### Params
    #     1. path (str) - filepath
    #     """
    #     self.model.save(path)

    # def load_model(self, path:str) -> None:
    #     """Load trained model from file.
        
    #     ### Params
    #     1. path (str) - filepath
    #     """
    #     self.model = models.load_model(path)
    