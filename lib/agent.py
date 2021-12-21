from lib.memory import Memory
from lib.policy import BoltzmannQPolicy, EpsilonGreedyPolicy, Policy
from keras import Model

import numpy as np

class Agent:
    """Implementation of a deep q-learning agent.
    """
    def __init__(self, model_factory, memory:Memory, policy:Policy,
                    learning_rate:float, discount_rate:float, update_rate:int) -> None:
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

        self.lr = learning_rate
        self.dr = discount_rate

        self.update_rate = update_rate
        self.update_cntr = 0

    def predict(self, state:np.ndarray, is_batch_predict=False) -> np.ndarray:
        """Predicts the weight for each action for a given state. 
        Can accept multiple states (batch prediction)
        1. Note that for batch prediction, please ensure that the states are
        wrapped in a np.ndarray along axis=0

        ### Params
        1. state: (np.ndarray) - np array of state or states
        2. is_batch_predict (bool) - whether to do batch_predicting or not. Note
        an incorrect specification will lead to a crash.

        ### Returns
        1. (np.ndarray) - weights for actions (either a singular weight vector or a batch of them)
        """
        if is_batch_predict == False:
            return self.target_model.predict(np.expand_dims(state, axis=0))[0]
        else:
            return self.target_model.predict(state)

    def apply_policy(self, action_weights:np.ndarray):
        """Outputs an action based off policy given action weights
        
        ### Params
        1. action_weights (np.ndarray of dtype=np.float32) - weights for every action
        """
        return self.policy.select(action_weights=action_weights)
    
    def remember(self, state:np.ndarray, action:np.int32, reward:np.float32, new_state:np.ndarray, is_done:bool, action_weights:np.ndarray) -> None:
        """Add new transition state to memory.
        ### Params
        1. state (np.ndarray) - current state
        2. action (np.int32) - action
        3. reward (np.float32) - reward value of action
        4. new_state (np.ndarray) - new state, may be empty
        5. is_done (bool) - true if new_state is a terminal state
        6. action_weights (np.ndarray) - weights for each action
        """
        self.memory.add_transition(state, action, reward, new_state, is_done, action_weights)

    def learn(self, batch_size:int=64) -> None:
        """Trains neural network (learning part)
        ### Params
        1. batch_size (int) - number of samples to take from memory
        """
        
        if self.memory.size + 2 <= batch_size:
            return

        # grab batch
        curr_states, actions, rewards, future_states, terminals, curr_qs, future_qs = self.memory.sample_transitions(sample_size=batch_size)
         
        X = []
        Y = []

        for idx in range(batch_size):
            current_state = curr_states[idx]
            action        = actions[idx]
            reward        = rewards[idx]
            done          = terminals[idx]
            
            # (reward + dr * Q(s_t+1, a_t) )
            if not done:
                best_future_q = np.max(future_qs[idx])
                new_q = reward + self.dr * best_future_q
            else:
                new_q = reward
            
            adjust_qs = curr_qs[idx]
            adjust_qs[action] = new_q

            X.append(current_state)
            Y.append(adjust_qs)
        self.model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0)

        self.update_cntr += 1
        if self.update_cntr >= self.update_rate:
            self.update_cntr = 0
            self.target_model.set_weights(self.model.get_weights())

        
    

def main():
    import gym

    from keras.models import Sequential
    from keras.layers import Dense, Input
    from keras.optimizer_v2.adam import Adam

    from tqdm import tqdm

    ### create environment
    env             = gym.make('CartPole-v1')
    observation     = env.reset()

    ### create model
    def factory():
        model = Sequential()
        model.add(Input(shape=env.observation_space.shape))
        model.add(Dense(256,                activation='relu'))
        model.add(Dense(256,                activation='relu'))
        model.add(Dense(256,                activation='relu'))
        model.add(Dense(env.action_space.n, activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(), metrics=['accuracy'])
        return model

    ### create agent
    agent = Agent(
        model_factory=factory,
        memory=Memory(input_dims=env.observation_space.shape, output_dims=(env.action_space.n, ) , max_len=1_000,),
        policy=BoltzmannQPolicy(temperature=1),
        learning_rate=0.01,
        discount_rate=0.95,
        update_rate=10
    )

    ### populate memory
    NUM_STEPS = 100_000
    for step in tqdm(range(NUM_STEPS)):
        action_weights = agent.predict(state=observation)
        action         = agent.apply_policy(action_weights=action_weights)
        new_obs, reward, done, info = env.step(action)
        
        env.render()

        if done:
            new_obs = env.reset()
            agent.learn(batch_size=64)

        agent.remember(observation, action, reward, new_obs, done, action_weights)

        observation = new_obs



    env.close()

    

if __name__ == '__main__':
    main()
        

    
    