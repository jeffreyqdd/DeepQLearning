from lib.memory import Memory
from lib.policy import Policy
from keras import Model

import numpy as np

class Agent:
    """Implementation of a deep q-learning agent.
    """
    def __init__(self, model:Model, memory:Memory, policy:Policy) -> None:
        """Constructor of the Agent class
        ### Params
        1. model  (Model)  - keras Model used to predict actions
        2. memory (Memory) - replay memory object from the library
        3. policy (Policy) - policy object from the library
        """
        self.memory = memory     
        self.policy = policy
        self.model  = model

    def predict(self, state:np.ndarray, is_batch_predict=False) -> None:
        """Predicts the weight for each action for a given state. 
        Can accept multiple states (batch prediction)
        1. Note that for batch prediction, please ensure that the states are
        wrapped in a np.ndarray along axis=0

        ### Params
        1. state: (np.ndarray) - np array of state or states
        2. is_batch_predict (bool) - whether to do batch_predicting or not. Note
        an incorrect specification will lead to a crash.
        """
        if is_batch_predict == False:
            return self.predict(np.expand_dims(a=state, axis=0))[0]
        else:
            return self.predict(state)
def main():
    import gym
    from keras.models import Sequential
    from keras.layers import Dense, Input
    from keras.optimizer_v1 import Adam

    ### create environment
    env             = gym.make('CartPole-v1')
    observation     = env.reset()

    ### create model
    model = Sequential()
    model.add(Input(shape=(env.observation_space.shape)))
    model.add(Dense(256,                activation='relu'))
    model.add(Dense(256,                activation='relu'))
    model.add(Dense(256,                activation='relu'))
    model.add(Dense(env.action_space.n, activation='linear'))
    model.summary()

        #     inputs = Input(shape=(n_inputs, ), name='state')
        # x = Dense(256, activation='relu')(inputs)
        # x = Dense(256, activation='relu')(x)
        # x = Dense(256, activation='relu')(x)
        # x = Dense(n_outputs,
        #           activation='linear', 
        #           name='action')(x)
        # q_model = Model(inputs, x)
        # self.q_model.compile(loss='mse', optimizer=Adam())


if __name__ == '__main__':
    main()
        

    
    