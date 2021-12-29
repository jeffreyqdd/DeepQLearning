from abc import ABC, abstractclassmethod

class Processor(ABC):
    def process_step(self, observation, reward, done, info):
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        return observation, reward, done, info
    
    @abstractclassmethod
    def process_observation(self, observation):
        pass
        
    @abstractclassmethod
    def process_reward(self, reward):
        pass

    @abstractclassmethod
    def process_info(self, info):
        pass

    @abstractclassmethod
    def process_action(self, action):
        pass

class LazyProcessor(Processor):
    def process_observation(self, observation):
        return observation
        
    def process_reward(self, reward):
        return reward

    def process_info(self, info):
        return info

    def process_action(self, action):
        return action
