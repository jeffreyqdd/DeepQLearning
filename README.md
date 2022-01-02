# Deep Q Learning
TODO: insert demos here

# About
My custom-developed library for deep q learning. Desgined based on OpenAI's gym environment, but with future, more complex projects in mind. I've dabbled in deep q learning in the past. However, this is my genuine attempt at learning and "mastering" deep Q learning.

The goal is to create a deep-q-learning library from scratch and from my own personal understanding, then learn from keras-rl. My goal is not to copy keras-rl and create a inferior version, but to learn what keras-rl does well and improve my own skills.

## Step 1
**Complete as of (Dec 21 2021)**
- I've successfully created a library that successfully trains a DQN-Agent to solve the environment 'CartPole-v1'. I wrote everything myself, pulling from my limited knowledge on deep-q-learning.
- After 5 hours of training, my model is still not bullet-proof. I think this has to do with my ad-hoc implementation. I'm sure there are more robust methods out there. 
- The next step is to improve upon the library. As of December 21, 2021, the original library is "archived" and will be deleted in future versions. 

**Goals**
- [x] Memory with random sampling
- [x] Policy (Epsilon, and Boltzmann)
- [x] DQN Agent
- [x] Test scripts


## Step2
**Complete as of (Jan 1 2022)**
- I am going to take some inspiration from the open-source library [keras-rl](https://github.com/keras-rl/keras-rl). The library is built using state-of-the-art deep reinforcement algorithms.
- Note: the library is difficult to understand.
- Update: Most of the features in the library are still beyond me. I will re-approach this later in my education. Hopefully, I will have the skillsets to successfully implement the features.
- As of January 1st 2022, the code is messy and needs to be properly documented/cleaned up.

**Goals**
- [x] Faster Memory Sampling
- [x] Uniform Memory
- [ ] Prioritized Memory (may implement in the near future)
- [x] Better Policies (Epsilon, Boltzmann)
- [x] Callbacks
- [x] DQN Agent
- [ ] Dueling DQN Agent (not going to implement)
- [x] Test scripts

## Step 2.5
**WIP as of (January 1 2022)**
- I've added an intermediate step. I need to go back and clean up my code, and add functionalities to make it more user friendly. Todolist is listed in goals.

**Goals**
- [ ] Clean up code
- [ ] Document code
- [ ] Create GUI for interactive visualizer.
- [ ] Streamline training code (through helper functions and smarter background processes)
- [ ] Implement OpenAI's classic control environments 
- [ ] Implement OpenAI's Box2D environments
- [ ] Implement OpenAI's atari enviroments (just a few. There are a lot).


## Step 3
**Not started as of (January 1 2022)**
- I am going to use my library to create board game AIs.


**Goals**
- [ ] Connect 2
- [ ] Connect 3
- [ ] Connect 4
- [ ] Connect 5
- [ ] Go
- [ ] Chess

# Library Development
## Documentation Stem
```python
def function1(foo) -> None:
    """Document me.
    ### Params:
        foo (type) - bar
    ### Returns:
        (type) - 
    """
    pass

class Class1(BaseClass):
    """Document me."""
    def __init__(self, foo) -> None:
        """Document me.
        ### Params:
            foo (type) - bar
        """
        pass
```

## Step 1 notes
Note: With deep q learning, only use reward, discount, and estimated future value.
![q-learning-formula](/assets/q-learning-formula.png)

## Step 2
### Developing the replay buffer:


### Developing the policies

### Developing the callbacks

### Developing the agent

# Setup:
**Originally done in Python 3.7.7**

**Made it work in Python 3.9.7 (requirements.txt reflects this)**

### Note
1. Virtual env creation will fail if swig is not installed

### Create the python virtual environment
1. ```python3 -m venv env```
2. ```source env/bin/activation```
3. ```pip install -r requirements.txt```

### Note that to create a requirements.txt file, do the following:
1. ```pip freeze > requirements.txt```

### To permanently add project root to python module search path 

1. ```cd $(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")```
2. ```echo /path/to/project/root > project-root.pth```