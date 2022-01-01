## pulled from https://github.com/keras-rl/keras-rl/blob/master/rl/core.py

from tensorflow.python.util.tf_decorator import rewrap
from core.callbacks import CallbackListHanlder
from abc import ABC, abstractclassmethod, abstractproperty
from keras.callbacks import History
from core.processor import LazyProcessor
from copy import deepcopy
import numpy as np


class BaseAgent(ABC): 
    """Abstract Base Class for Agent.
    
    Implement the following methods/properties for your own agent
    - forward
    - backward
    - compile
    - load_weights
    - save_weights
    - layers

    Every agent contains the following callbacks during the fit "training" process. Can be overridden. 
    - _on_train_begin
    - _on_train_end
    - _on_test_begin
    - _on_test_end

    """

    def __init__(self, processor=None):
        self.step = 0
        self.processor = processor if (processor is not None) else LazyProcessor()
    
    def get_config(self):
        return {}

    def fit(self, 
        env,
        nb_steps,
        nb_max_episode_steps=None, 
        action_repetition=1, 
        callbacks=[], 
    ):
        """ Fits the model to the environment.

        Returns a keras History obj
        """
        
        #########################################################################################
        ############# Initialize
        #########################################################################################
        history     = History()
        callbacks   += [history] # add keras history callback
        callbacks   = CallbackListHanlder(callbacks)

        callbacks.set_model(self)
        callbacks.set_env(env)

        #########################################################################################
        ############# On Train Begin
        #########################################################################################
        self._on_train_begin()
        callbacks.on_train_begin()

        episode = 0
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False
        
        # allow keyboard interrupt to abort training process
        try:
            while self.steps < nb_steps:
                # reset environment if observation is None
                if observation is None:
                    #########################################################################################
                    ############# On Episode Begin
                    #########################################################################################
                    callbacks.on_episode_begin(episode)
                    
                    episode_step = 0
                    episode_reward = 0

                    self.reset_states() # reset internal memory built up during a single episode

                    observation = deepcopy(env.reset())
                    observation = self.processor.process_observation(observation)         

                # expect to be fully initialized
                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None

                #########################################################################################
                ############# On Step Begin
                #########################################################################################
                callbacks.on_step_begin(episode_step)

                # forward
                step_action = self.forward(observation)
                step_action = self.processor.process_action(step_action)
                
                step_reward = 0
                step_done = False
                step_info = {}
                new_observation = None
                
                # repeat action_repetition
                for _ in range(action_repetition):
                    #########################################################################################
                    ############# On Action Begin
                    #########################################################################################
                    callbacks.on_action_begin(step_action)
                    new_observation, r, step_done, info = env.step(step_action)
                    new_observation = deepcopy(new_observation)

                    new_observation, r, step_done, info = self.processor.process_step(new_observation, r, step_done, info)
                    new_observation = deepcopy(new_observation)

                    # record real numbers in step_info
                    for key, value in info.items():
                        if np.isreal(value):
                            if key not in step_info:
                                step_info[key] = value
                            else:
                                step_info[key] += value
                    step_reward += r
                    #########################################################################################
                    ############# On Action End
                    #########################################################################################
                    callbacks.on_action_end(step_action)
                    # could be done while doing repition
                    if step_done:
                        break
                    
                # force terminal step if "solved"
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    step_done = True
                
                metrics = self.backward(
                    observation,
                    step_action,
                    step_reward,
                    new_observation,
                    step_done,
                )
                
                # logs
                step_logs = {
                    'action' : step_action,
                    'observation' : observation,
                    'reward' : step_reward,
                    'metrics' : metrics, 
                    'episode' : episode,
                    'info' : step_info, 
                }
                #########################################################################################
                ############# On Step End
                #########################################################################################
                callbacks.on_step_end(episode_step, step_logs)

                observation     = new_observation
                episode_reward  += step_reward
                
                episode_step    += 1
                self.step       += 1
                
                if step_done:
                    #########################################################################################
                    ############# On Episode End
                    #########################################################################################
                    episode_logs = {
                        'episode_reward': episode_reward,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.step,
                    }
                    callbacks.on_episode_end(episode, episode_logs)

                    episode += 1
                    observation = None
                    episode_step = None
                    episode_reward = None

        except KeyboardInterrupt:
            did_abort = True

        #########################################################################################
        ############# On Train End
        #########################################################################################
        callbacks.on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()

        return history



    def reset_states(self):
        """Resets all internally kept states after an episode is completed.
        """
        pass

    def forward(self, observation):
        """Takes the an observation from the environment and returns the action to be taken next.
        If the policy is implemented by a neural network, this corresponds to a forward (inference) pass.

        # Argument
            observation (object): The current observation from the environment.

        # Returns
            The next action to be executed in the environment.
        """
        raise NotImplementedError()

    def backward(self, state, action, reward, new_state, terminal):

        raise NotImplementedError()

    def compile(self, optimizer, metrics=[]):
        """Compiles an agent and the underlaying models to be used for training and testing.

        # Arguments
            optimizer (`keras.optimizers.Optimizer` instance): The optimizer to be used during training.
            metrics (list of functions `lambda y_true, y_pred: metric`): The metrics to run during training.
        """
        raise NotImplementedError()

    def load_weights(self, filepath):
        """Loads the weights of an agent from an HDF5 file.

        # Arguments
            filepath (str): The path to the HDF5 file.
        """
        raise NotImplementedError()

    def save_weights(self, filepath, overwrite=False):
        """Saves the weights of an agent as an HDF5 file.

        # Arguments
            filepath (str): The path to where the weights should be saved.
            overwrite (boolean): If `False` and `filepath` already exists, raises an error.
        """
        raise NotImplementedError()

    @property
    def layers(self):
        """Returns all layers of the underlying model(s).

        If the concrete implementation uses multiple internal models,
        this method returns them in a concatenated list.

        # Returns
            A list of the model's layers
        """
        raise NotImplementedError()

    @property
    def metrics_names(self):
        """The human-readable names of the agent's metrics. Must return as many names as there
        are metrics (see also `compile`).

        # Returns
            A list of metric's names (string)
        """
        return []

    def _on_train_begin(self):
        """Callback that is called before training begins."
        """
        pass

    def _on_train_end(self):
        """Callback that is called after training ends."
        """
        pass

    def _on_test_begin(self):
        """Callback that is called before testing begins."
        """
        pass

    def _on_test_end(self):
        """Callback that is called after testing ends."
        """
        pass