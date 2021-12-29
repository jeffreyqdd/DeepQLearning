## pulled from https://github.com/keras-rl/keras-rl/blob/master/rl/callbacks.py
from keras.callbacks import Callback
from numpy import np
import warnings
import timeit

class BaseCallback(Callback):
    """Abstract Callback class.

    Implements the following methods than can be overidden:
    - on_episode_begin
    - on_episode_end
    - on_step_begin
    - on_step_end
    - on_action_begin
    - on_action_end
    """
    def set_env(self, env):
        """Set the environment for internal reference"""
        self.env = env

    def on_episode_begin(self, episode, logs={}):
        """Called at beginning of each episode"""
        pass

    def on_episode_end(self, episode, logs={}):
        """Called at end of each episode"""
        pass

    def on_step_begin(self, step, logs={}):
        """Called at beginning of each step"""
        pass

    def on_step_end(self, step, logs={}):
        """Called at end of each step"""
        pass

    def on_action_begin(self, action, logs={}):
        """Called at beginning of each action"""
        pass

    def on_action_end(self, action, logs={}):
        """Called at end of each action"""
        pass

class CallbackListHanlder:
    """Provides simple 'vectorized' method calls on a list of keras.callbacks.Callback objects

    Note: Each Callback object also inherits properties from keras.callbacks.Callback.
    Thus if BaseCallback methods are not available, we must call keras callback methods instead
    """

    def __init__(self, callbacks):
        self.callbacks = callbacks

    def set_env(self, env):
        for callback in self.callbacks:
            if callable(getattr(callable, 'set_env', default = None)):
                callback.set_env(env)
    
    def set_model(self, model):
        for callback in self.callbacks:
            if callable(getattr(callable, 'set_model', default = None)):
                callback.set_model(model)

    def on_episode_begin(self, episode, logs={}):
        for callback in self.callbacks:
            if callable(getattr(callable, 'on_episode_begin', default = None)):
                callback.on_episode_begin(episode, logs)
            else: # default to built-in-keras terminology
                callback.on_epoch_begin(episode, logs=logs)

    def on_episode_end(self, episode, logs={}):
        for callback in self.callbacks:
            if callable(getattr(callable, 'on_episode_end', default = None)):
                callback.on_episode_end(episode, logs)
            else: # default to built-in-keras terminology
                callback.on_epoch_end(episode, logs=logs)

    def on_step_begin(self, step, logs={}):
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_step_begin', None)):
                callback.on_step_begin(step, logs)
            else: # default to built-in-keras terminology
                callback.on_batch_begin(step, logs)

    def on_step_end(self, step, logs={}):
        for callback in self.callbacks:
            if callable(getattr(callable, 'on_step_end', default = None)):
                callback.on_step_end(step, logs)
            else: # default to built-in-keras terminology
                callback.on_batch_end(step, logs)

    def on_action_begin(self, action, logs={}):
        for callback in self.callbacks:
            if callable(getattr(callable, 'on_action_begin', default = None)):
                callback.on_action_begin()


    def on_action_end(self, action, logs={}):
        for callback in self.callbacks:
            if callable(getattr(callable, 'on_action_end', default = None)):
                callback.on_action_end(action, logs)

class MetricsCallback(BaseCallback):
    def __init__(self, do_free_resources=False):
        # for clean implementation we will store data by episodes
        self.episode_start  = {}
        self.observations   = {}
        self.rewards        = {}
        self.actions        = {}
        self.metrics        = {}
        self.step           = 0

        self.do_free_resources = do_free_resources

    def on_train_begin(self, logs=None):
        """Training values"""
        self.train_start = timeit.default_timer()
        self.metric_names = self.model.metrics_names
        print(f"Training for {self.params['nb_steps']} steps ...")

    def on_train_end(self, logs=None):
        """Print training time"""
        duration = timeit.default_timer() - self.train_start
        print(f'done, took {duration:.3f} seconds')

    def on_episode_begin(self, episode, logs={}):
        self.episode_start[episode] = timeit.default_timer()
        self.observations[episode]  = []
        self.rewards[episode]       = []
        self.actions[episode]       = []
        self.metrics[episode]       = []
    
    def on_episode_end(self, episode, logs={}):
        """Print episode training statistics"""
        duration = timeit.default_timer() - self.episode_start[episode]
        episode_steps = len(self.observations[episode])

        # Format all metrics.
        metrics = np.array(self.metrics[episode])
        metrics_template = ''
        metrics_variables = []
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for idx, name in enumerate(self.metrics_names):
                if idx > 0:
                    metrics_template += ', '
                try:
                    value = np.nanmean(metrics[:, idx])
                    metrics_template += '{}: {:f}'
                except Warning:
                    value = '--'
                    metrics_template += '{}: {}'
                metrics_variables += [name, value]          
        metrics_text = metrics_template.format(*metrics_variables)

        nb_step_digits = str(int(np.ceil(np.log10(self.params['nb_steps']))) + 1)
        template = '{step: ' + nb_step_digits + 'd}/{nb_steps}: episode: {episode}, duration: {duration:.3f}s, episode steps: {episode_steps:3}, steps per second: {sps:3.0f}, episode reward: {episode_reward:6.3f}, mean reward: {reward_mean:6.3f} [{reward_min:6.3f}, {reward_max:6.3f}], mean action: {action_mean:.3f} [{action_min:.3f}, {action_max:.3f}],  {metrics}'
        variables = {
            'step': self.step,
            'nb_steps': self.params['nb_steps'],
            'episode': episode + 1,
            'duration': duration,
            'episode_steps': episode_steps,
            'sps': float(episode_steps) / duration,
            'episode_reward': np.sum(self.rewards[episode]),
            'reward_mean': np.mean(self.rewards[episode]),
            'reward_min': np.min(self.rewards[episode]),
            'reward_max': np.max(self.rewards[episode]),
            'action_mean': np.mean(self.actions[episode]),
            'action_min': np.min(self.actions[episode]),
            'action_max': np.max(self.actions[episode]),
            'metrics': metrics_text
        }
        print(template.format(**variables))

        # Free up resources.
        if self.do_free_resources:
            del self.episode_start[episode]
            del self.observations[episode]
            del self.rewards[episode]
            del self.actions[episode]
            del self.metrics[episode]
    
    def on_step_end(self, step, logs):
        """ Update statistics of episode after each step """
        episode = logs['episode']

        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        self.metrics[episode].append(logs['metrics'])
        
        self.step += 1


class VisualizerCallback(BaseCallback):
    def on_action_end(self, action, logs={}):
        """Show Training"""
        self.env.render()