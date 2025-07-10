# Udacity DRL - Project 2: Continuous Control
### Introduction

The objective of the agent is to learn a policy that moves and maintains the double-jointed arm’s hand at the target location for as many time steps as possible to maximize cumulative reward.

![](./img/reacher.gif)

### The environment
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training
This project includes two environments, and my solution focuses on Option 2.

- The first version contains a single agent.

- The second version contains 20 identical agents, each with its own copy of the environment.

The second version is useful for algorithms like PPO(opens in a new tab), A3C(opens in a new tab), and D4PG(opens in a new tab) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.

### Solving Criteria for option 2

The agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
This yields an average score for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. In the case of the plot above, the environment was solved at episode 63, since the average of the average scores from episodes 64 to 163 (inclusive) was greater than +30.

### Installation
1. Clone the Course Repository and set up the Python environment instructions in the DRLND GitHub repository.

2. Download the Unity environment:

- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- [Windows 32-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- [Windows 64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

3. Place the unzipped folder from step 2 in p2_continuous-control/ folder (the DRLND GitHub repository).

### How to run the code
You can train and test the agent using the Continuous_Control.ipynb notebook. Follow the instructions provided in the notebook to adjust the hyperparameters as needed

