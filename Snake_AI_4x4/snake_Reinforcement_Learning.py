import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import time
from collections import deque, namedtuple
from snake_objects import Board, rewards
from snake_utils import agent_learn, get_experiences

# Constants
MEMORY_SIZE = 2000     # size of memory buffer
GAMMA = 0.995          # discount factor
NUM_STEPS_FOR_UPDATE = 1  # perform a learning update in every step
E_DECAY = 0.995        # ε-decay rate for the ε-greedy policy
E_MIN = 0.01           # Minimum ε value for the ε-greedy policy

# Game parameter
NUM_ACTIONS = 4
BOARD_SIZE = 4
board_width = BOARD_SIZE

# Training variables
minibatch_Size = 16     # Mini-batch size
num_games = 400       # number of games to run
num_games_avg = 100    # number of latest game scores to use for averaging
epsilon = 1.0          # initial ε value for ε-greedy policy

max_tries = BOARD_SIZE * BOARD_SIZE * 2
final_score_history = []
num_steps_history = []

# Q-Network to model the Q-values of the states
q_network = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(board_width ** 2,)),
    tf.keras.layers.Dense(72, activation='swish'),
    tf.keras.layers.Dense(72, activation='swish'),
    tf.keras.layers.Dense(NUM_ACTIONS, activation='linear')
])
print(q_network.summary())

# Mirror the Q-Network for soft update
target_q_network = tf.keras.models.clone_model(q_network)

"""
Store experiences as named tuples. The current board values is the current
state, the direction in which the snake moves is the action, and the board
values got by taking the action is the next state.
Game termination status is saved as done values.
"""
experience = namedtuple("Experience", field_names=["state", "action", "reward",
                                                   "next_state", "done"])

start = time.time()

# Create a memory buffer with capacity = MEMORY_SIZE
memory_buffer = deque(maxlen=MEMORY_SIZE)

# Set the target-Q-Network weights equal to the Q-Network weights
target_q_network.set_weights(q_network.get_weights())

high_score = 1

# Run as many games as possible for the agent to learn
for game in range(num_games):
    # Reset the environment to the initial state and get the initial state
    new_board = Board(BOARD_SIZE)
    board_values = new_board.board_values
    final_score = 0
    step = 0
    dry_step = 0

    # Increase minibatch size as more experience tuples are available
    if game == 50:
        minibatch_Size = 32
    if game == 100:
        minibatch_Size = 64

    while not new_board.game_over:
        step += 1

        # From the current state S choose an action A using an ε-greedy policy
        state_qn = np.reshape(board_values, (1, -1))
        q_values = q_network(state_qn)
        # Select randomly between exploration and exploitation
        sel_r = random.random() > epsilon
        direction = sel_r * np.argmax(q_values, axis=1) +\
            (1 - sel_r) * random.choice(range(NUM_ACTIONS))

        # Take action A and receive reward R and the next state S'
        next_board_values, reward, game_over = new_board.next_move(direction)

        # Store experience tuple (S, A, R, S') in the memory buffer
        # We store the game termination status as well for convenience
        memory_buffer.append(experience(board_values, direction, reward,
                                        next_board_values, game_over))

        # Only update the network every NUM_STEPS_FOR_UPDATE time steps
        if step % NUM_STEPS_FOR_UPDATE == 0 and \
                len(memory_buffer) > minibatch_Size:
            # Sample random mini-batch of experience tuples from memory buffer
            experiences = get_experiences(memory_buffer, minibatch_Size)

            # Perform a gradient descent step and update the network weights
            agent_learn(experiences, GAMMA, q_network, target_q_network)

        board_values = next_board_values.copy()
        final_score = new_board.score
        if final_score > high_score:
            high_score = final_score

        # Count dry steps
        if reward != rewards["food"]:
            dry_step += 1
        else:
            dry_step = 0

        if game_over or dry_step >= max_tries:
            break

    final_score_history.append(final_score)
    num_steps_history.append(step)
    avg_latest_score = np.mean(final_score_history[-num_games_avg:])

    # Update the ε value
    epsilon = max(E_MIN, epsilon * E_DECAY)

    if (game + 1) % num_games_avg == 0:
        print(f"\nGame {game+1}")
        print(f"Average score of the last {num_games_avg} games: {avg_latest_score:.2f}")
        print(f"High score: {high_score} | Epsilon: {epsilon:.4f}")

    # We will consider that the environment is solved if we can score
    # average of 75% of maximum possible score in the last 100 games
    if avg_latest_score >= (BOARD_SIZE ** 2) * 0.75:
        print(f"\nAgent optimized in {game+1} games!")
        break

q_network.save('snake.keras')
tot_time = time.time() - start

print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time / 60):.2f} min)")

plt.figure()
plt.stem(final_score_history, markerfmt=" ")
plt.xlabel("Game No.")
plt.ylabel("Score")
plt.title("Score history")

plt.figure()
plt.stem(num_steps_history, markerfmt=" ")
plt.xlabel("Game No.")
plt.ylabel("No. of steps")
plt.title("Number of steps")

plt.show()
