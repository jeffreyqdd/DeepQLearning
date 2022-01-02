from keras.models import clone_model
from core.memory import Memory
from core.policy import BoltzmannQPolicy, EpsilonGreedyQPolicy
from core.agent import BaseAgent

from keras import Model
import keras.backend as K
from keras.layers import Lambda


import numpy as np

class DqnAgent(BaseAgent):
    """Implementation of a deep q-learning agent.
    """
    def __init__(self, model, memory, policy, discount_rate, nb_warmup_steps, target_update_rate, train_every, batch_size, *args, **kwargs):

        """Constructor of the Agent class
        ### Params
        1. model  (function)  - function with no arguments that returns a keras Model
        2. memory (Memory) - replay memory object from the library
        3. policy (Policy) - policy object from the library
        4. learning_rate (float) - learning rate of the deep q agent
        5. discount_rate (float) - discount rate of the deep q agent
        6. update_rate (int) - update frequency of target model from learn calls
        """
        super().__init__(*args, **kwargs)

        self.memory = memory     
        self.policy = policy
        self.model:Model  = model
        self.target_model:Model = None

        self.gamma = discount_rate

        self.nb_warmup_steps = nb_warmup_steps

        self.target_model_update = target_update_rate
        self.update_cntr = 0

        self.train_every = train_every
        self.batch_size = batch_size

    def forward(self, observation):
        action = self.predict_on_observation(observation)
        return self.policy.select(action_weights=action)
    

    def backward(self, state, action, reward, new_state, terminal):
        self.memory.append(state, action, reward, new_state, terminal)

        if self.step < self.nb_warmup_steps:
            return
        if self.step % self.train_every != 0:
            return

        X = []
        Y = []
        
        experiences = self.memory.sample(self.batch_size)

        state0    = [None] * self.batch_size
        action0   = [None] * self.batch_size
        reward1   = [None] * self.batch_size
        state1    = [None] * self.batch_size
        terminal1 = [None] * self.batch_size

        for idx, transition in enumerate(experiences):
            state0[idx]     = transition.state0
            action0[idx]    = transition.action
            reward1[idx]    = transition.reward
            state1[idx]     = transition.state1
            terminal1[idx]  = transition.terminal1

        # batch predict
        current_qs = self.predict_on_batch(np.array(state0))
        future_qs = self.predict_on_batch(np.array(state1))

        for idx in range(self.batch_size):
            # reward + dr * Q(state1, action0)

            if not terminal1[idx]:
                best_future_q = np.max(future_qs[idx])
                new_q = reward + self.gamma * best_future_q
            else:
                new_q = reward
            
            adjust_qs = current_qs[idx]
            adjust_qs[action0[idx]] = new_q

            X.append(state0[idx])
            Y.append(adjust_qs)
        
        X = np.array(X)
        Y = np.array(Y)

        # raise KeyboardInterrupt()
        self.model.fit(X, Y, verbose=0)

        self.update_cntr += 1
        if self.update_cntr >= self.target_model_update:
            self.update_cntr = 0
            self.target_model.set_weights(self.model.get_weights())
    
    def compile(self, optimizer, loss, metrics=[]):

        # we never train the target model, so we pass random arguments
        self.target_model = clone_model(self.model)
        self.target_model.compile(optimizer='sgd', loss='mse')

        self.model.compile(optimizer, loss, metrics)


    def predict_on_observation(self, observation):
        return self.target_model.predict(np.expand_dims(observation, axis=0))[0]

    def predict_on_batch(self, observations):
        return self.target_model.predict(observations)

    def save_weights(self, filepath, overwrite=False):
        self.model.save(filepath=filepath, overwrite=overwrite)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.target_model.set_weights(self.model.get_weights())