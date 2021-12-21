# Deep Q Learning

## About
My custom-developed library for deep q learning. Designed based on OpenAI's gym environment.

I've dabbled in deep q learning in the past. However, this is my genuine attempt at learning and "mastering" deep Q learning.

## Goals
I want to apply what I learn here to more complex AI's e.g. tic tac toe, connect4, chess!

# Library Development (lib)
wip

note that the current implementation is done in tensorflow

I want future implementations to be doen in pytorch


# Setup:
**Note that I am using Python 3.7.7**

### Create the python virtual environment
1. ```python3 -m venv env```
2. ```source env/bin/activation```
3. ```pip install -r requirements.txt```

### Note that to create a requirements.txt file, do the following:
1. ```pip freeze > requirements.txt```

### To permanently project root to python module search path 

1. ```cd $(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")```
2. ```echo /path/to/project/root > project-root.pth```

