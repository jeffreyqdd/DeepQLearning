import gym

from lib.agent import Agent
from lib.tools import EnvWrapper
from lib.memory import Memory
from lib.policy import BoltzmannQPolicy

from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizer_v2.adam import Adam


def model_factory():
    model = Sequential()
    model.add(Input(shape= (4,) ) )
    model.add(Dense(16,                activation='relu'))
    model.add(Dense(16,                activation='relu'))
    model.add(Dense(16,                activation='relu'))
    model.add(Dense(2,                 activation='linear'))
    model.compile(loss='mse', optimizer=Adam(), metrics=['accuracy'])



    return model
    
if __name__ == '__main__':
    ### create agent
    agent = Agent(
        model_factory=model_factory,
        memory=Memory(input_dims=(4,), output_dims=(2,), max_len=10_000),
        policy=BoltzmannQPolicy(temperature=1.0),
        learning_rate=0.01,
        discount_rate=0.95,
        update_rate=10
    )

    ### create environment
    env             = gym.make('CartPole-v1')

    ### Train
    EnvWrapper.run(
        agent=agent,
        env=env,
        batch_size=128,
        warmup_steps=300,
        training_steps=500_000,
        save_every=20_000,
        load_dir=None,
        save_dir=None
    )


    

