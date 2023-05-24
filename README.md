# Snake Reinforcement Learning

In this project, I developed an AI model using reinforcement learning to play the arcade game snake. This is a self-initiative project to build my understanding of deep reinforcement learning and gain intuitions. Hence, the state values are taken as it is from the board without making any interpretation before passing them into the network and letting the network learn by itself.

## The Game
The game starts with a rectangular board with a snake and food randomly placed on the board. The snake moves around the board to catch the food. The snake initially occupies one square and grows in size every time it catches the food. The game score is the length of the snake. The goal is to score high by collecting as much food as possible without running into walls or its own body, which kills the snakes and terminates the game.

## Model
The model takes the game's state as the input, estimates the value function for all possible actions using a deep neural network, and selects the action with the best returns. Experiences are stored in a memory buffer and are sampled in mini-batches to update weights. It also uses a parallel target network with the same architecture and initial weights for soft updates. The model uses the ε-greedy policy to have a trade-off between exploration and exploitation.

## Contents
I started with smaller and simpler problems to gain intuition in developing neural networks and gradually increased the complexity and size.
* __Fully connected neural network__: In the first step, a 4x4 snake board is solved using a fully connected neural network. A numerical value is defined for each object (food, head, body, tail, wall, and nothing) and assigned to corresponding squares on the board. The board values are the inputs of the neural network. The network scores an average of 8.29 out of 16.
* __Convolutional neural network__: The same 4x4 board is solved using a convolutional neural network. The objects are assigned a vector of three values. The board values contain three channels used as RGB values for animation. They are the input to the convolutional neural network. The network scores an average of 12.01 out of 16.
* __Further tunning for larger boards__: With increasing the board size, possible states increase exponentially, and the board becomes sparse. The chance of the snake catching the food in the initial stages decreases significantly. Hence intermediate rewards/penalties are awarded for moving closer to/away from the food. The ε value is reduced linearly at a slower rate to explore more actions in the initial stages. For a 6x6 board, the network scores an average of 12.09 out of 36, with the highest score of 23. 
* Better designs for 6x6 and larger boards are in progress.

## Usage
Each subfolder contains the following files:
* _snake_reinforcement_learning.py_: Main file to perform simulations for reinforcement learning. Define the board dimension and network architecture here.
* _snake_objects.py_: Contains the class definitions for the snake game and associated methods.
* _snake_utils.py_: Contains functions to sample experience and updates weights.
* _snake_game_AI.py_: Run this file to see AI play the snake game.
* _snake.keras_: stores the trained neural network.

## Requirements
Following libraries are used for developing the project. Please use the below versions or versions compatible with them:
* Python 3.11.1
* Tesorflow 2.12.0
* Numpy 1.23.5
* Matplotlib 3.7.0

## Credits
Inspired by the "Machine Learning Specialization" course by Deeplearning.AI in Coursera and various online sources for training the game with reinforcement learning. Experimented with own definitions for reward functions, state values, etc., and designed the network architecture accordingly.

### Work done by Pranesh Raghavendran

