from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizer_v2.adam import Adam

import gym

env = gym.make('CartPole-v1')
    
    
def factory():
    model = Sequential()
    model.add(Input(shape=env.observation_space.shape))
    model.add(Dense(24,                activation='relu'))
    model.add(Dense(24,                activation='relu'))
    model.add(Dense(env.action_space.n, activation='linear'))
    model.summary()
    model.compile(loss='mse', optimizer=Adam(), metrics=['accuracy'])
    return model


from core.agents.dqn import DqnAgent
from core.memory import UniformMemory
from core.policy import BoltzmannQPolicy
from core.callbacks import VisualizerCallback, MetricsCallback
agent = DqnAgent(
    model_factory=factory,
    memory=UniformMemory(10000),
    policy=BoltzmannQPolicy(),
    discount_rate=0.99,
    nb_warmup_steps=1000,
    target_update_rate=10,
    train_every=1,
)

agent.fit(env, 20_000, 300, callbacks=[VisualizerCallback, MetricsCallback])