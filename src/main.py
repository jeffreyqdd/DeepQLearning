from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizer_v2.adam import Adam

from core.agents.dqn import DqnAgent
from core.processor import LazyProcessor
from core.memory import UniformMemory
from core.policy import EpsilonGreedyQPolicy, BoltzmannQPolicy
from core.callbacks import VisualizerCallback, MetricsCallback

import gym


env = gym.make('CartPole-v1')


def model_factory() -> Model:
    input_shape  = env.observation_space.shape
    output_shape = env.action_space.n
    
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(output_shape,  activation='linear'))
    return model


model = model_factory()
model.summary()

agent = DqnAgent(
    model=model,
    memory = UniformMemory(max_size=10_000),
    policy = BoltzmannQPolicy(),
    discount_rate= 0.99,
    nb_warmup_steps= 65,
    target_update_rate=4,
    train_every=1,
    batch_size=64,
    processor = LazyProcessor(),
)

agent.compile( optimizer='adam', loss='mse', metrics=['accuracy'] )

agent.fit(env, 30_000, 300, callbacks=[MetricsCallback()] )

agent.save_weights('bin/cartpole.h5', overwrite=False)
agent.load_weights('bin/cartpole.h5')

agent.test(env, nb_episode=10)

