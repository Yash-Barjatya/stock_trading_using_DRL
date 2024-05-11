# STOCK TRADING USING DEEP REINFORCEMENT LEARNING #

## Code Demo

[code-demo](https://drive.google.com/file/d/1CNsURGgGA9Z4fHPivX9-hxqwG01hB-HF/view?usp=sharing)

## Problem Statement

Prepare an agent by implementing Deep Q-Learning that can perform unsupervised trading in stock trade. The aim of this project is to train an agent that uses Q-learning and neural networks to predict the profit or loss by building a model and implementing it on a dataset that is available for evaluation.

## APPROACH

The stock trading index environment provides the agent with a set of actions:

- Buy
- Sell 
- Hold (Do Nothing)

The notebook has following sections:

- Import libraries
- Create a DQN agent
- Preprocess the data 
- Train and build the model 
- Evaluate the model and agent 


**Steps to perform**

In the section create a DQN agent, create a class called agent where:

Action size is defined as 3. <br>
Experience replay memory to deque is 1000. <br>
Empty list for stocks that has already been bought. <br>
The agent must possess the following hyperparameters: 
- gamma= 0.95
- epsilon = 1.0
- epsilon_final = 0.01
- epsilon_decay = 0.995 <br>


Neural network has 3 hidden layers <br>
Action and experience replay are defined <br>

**Code and Dataset** <br>
-   This repo contains python notebook whose each cells can be executed sequentially to define, train and test our model.
-   Dataset contains GSPC data in a csv file of different years.
-   Uploaded model weights were trained on GSPC_2014 data with 7 episodes.
