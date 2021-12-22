# Deep Q Learning
TODO: insert demos here

# About
My custom-developed library for deep q learning. Desgined based on OpenAI's gym environment, but with future, more complex projects in mind. I've dabbled in deep q learning in the past. However, this is my genuine attempt at learning and "matering" deep Q learning. 

**(Dec 21 2021) STEP 1 complete**
- I've successfully created a library that successfully trains a DQN-Agent to solve the environment 'CartPole-v1'. I wrote everything myself, pulling from my limited knowledge on deep-q-learning.
- After 5 hours of training, my model is still not bullet-proof. I think this has to do with my ad-hoc implementation. I'm sure there are more robust methods out there. 
- The next step is to improve upon the library. As of December 21, 2021, the original library is "archived" and will be deleted in future versions. 

**(No Date Yet) STEP 2 WIP**
- I am going to take heavy inspiration from the open-source library [keras-rl](https://github.com/keras-rl/keras-rl). The library is built using state-of-the-art deep reinforcement algorithms.

**(No Date Yet) STEP 3 WIP**
- I am going to use my library to create board game AIs.

# Goals
**STEP 1**
- [x] Memory with random sampling
- [x] Policy (Epsilon, and Boltzmann)
- [x] DQN Agent
- [x] Test scripts

**STEP 2**
- [ ] Faster Memory Sampling
- [ ] Sequential Memory
- [ ] Better Policy (Epsilon, Boltzmann)
- [ ] Callbacks
- [ ] Dueling DQN Agent
- [ ] Test scripts

**STEP 3**
- [ ] Connect 2
- [ ] Connect 3
- [ ] Connect 4
- [ ] Connect 5
- [ ] Go
- [ ] Chess

# Library Development
## Documentation Stem
```python
def function1(foo):
    """Document me.
    ### Params:
        foo (type) - bar
    ### Returns:
        (type) - 
    """
    pass
```

## Step 1 notes
Note: With deep q learning, only use reward, discount, and estimated future value.
![q-learning-formula](/assets/q-learning-formula.png)

## Step 2
### Memory
```python
class ddfd:
    pass
```

# Setup:
**Done in Python 3.7.7**

### Create the python virtual environment
1. ```python3 -m venv env```
2. ```source env/bin/activation```
3. ```pip install -r requirements.txt```

### Note that to create a requirements.txt file, do the following:
1. ```pip freeze > requirements.txt```

### To permanently project root to python module search path 

1. ```cd $(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")```
2. ```echo /path/to/project/root > project-root.pth```

# Resources
*insert links to sources here*