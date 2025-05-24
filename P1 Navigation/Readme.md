# Project 1: Navigation

## Introduction

The objective of this porject is to train an agent to navigate (and collect bananas!) in a large, square world.

![](./img/Rl_collect_banana.gif)

##  The environment

The project environment is similar to, but not identical to the Banana Collector environment on the Unity ML-Agents GitHub page  [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents).

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- `0` - move forward
- `1` - move backward
- `2` - turn left
- `3` - turn right

### Solving Criteria 

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Installation

**1.** Clone the Course Repository and set up the Python environment [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies). 

**2.** Download the Unity environment:
- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- [Windows 32-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- [Windows 64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

**3.** Place the unzipped folder from step 2 in  `p1_navigation/` folder (the DRLND GitHub repository).

## How to run the code

You can train and test the agent using the Navigation.ipynb notebook. Follow the instructions provided in the notebook to adjust the hyperparameters as needed