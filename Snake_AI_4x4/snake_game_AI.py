import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from snake_objects import Board

# Define board size
BOARD_SIZE = 4
board_width = BOARD_SIZE + 2
max_tries = BOARD_SIZE * BOARD_SIZE * 2

# Create new board and start the animation
new_board = Board(BOARD_SIZE)
im = new_board.draw_board()

# Load the Q-Network model
file_name = "snake.keras"
q_network = tf.keras.saving.load_model(file_name)

print("Snake game using AI")
dry_step = 0
step = 0

while (not new_board.game_over) and (dry_step <= max_tries):
    # Choose the action using Q-Network
    state_qn = np.reshape(new_board.board_values, (1, -1))
    q_values = q_network(state_qn)
    direction = np.argmax(q_values, axis=1)

    # Next move
    old_score = new_board.score
    new_board.next_move(direction)
    new_score = new_board.score

    im = new_board.draw_board(im)

    # Count steps
    if old_score == new_score:
        dry_step = dry_step + 1
    else:
        dry_step = 0
    step = step+1

print("Score: ", new_board.score)
plt.show()
