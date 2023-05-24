import numpy as np
import matplotlib.pyplot as plt
import random

# Pixel values for each type of object
board_objects = {
    "none": np.array([0, 0, 0]),
    "food": np.array([1, 0, 0]),
    "tail": np.array([1, 1, 0]),
    "head": np.array([1, 1, 1]),
    "body": np.array([0, 1, 0]),
    "wall": np.array([0, 0, 1])
}

# Rewards for every outcome
rewards = {
    "food": 10,
    "hit": -10,
    "none": 0
}


class Board:
    """
    Board object stores the board state and methods to initialize board, place
    new food, update board state, make a move and display the board
    """
    def __init__(self, board_size):
        """
        Randomly initialize the board with snake of length 1 and a food
        """
        self.board_size = board_size
        self.direction = random.randint(0, 4)
        self.X = [random.randint(1, board_size)]
        self.Y = [random.randint(1, board_size)]
        self.XY = [(self.X[i], self.Y[i]) for i in range(len(self.X))]
        self.score = len(self.XY)
        self.fx, self.fy, self.food = self.get_new_food()
        self.board_values = self.get_board_vals()
        self.reward = 0
        self.game_over = 0

    def get_new_food(self):
        """
        Randomly place food in a empty square
        """
        while True:
            fx = random.randint(1, self.board_size)
            fy = random.randint(1, self.board_size)
            fxy = (fx, fy)
            if fxy not in self.XY:
                break
        return fx, fy, fxy

    def get_board_vals(self):
        """
        Fill the board with values from the dictionary board_objects
        """
        board_values = np.zeros((self.board_size + 2, self.board_size + 2, 3))
        board_values[0, :, :] = board_objects["wall"]
        board_values[-1, :, :] = board_objects["wall"]
        board_values[:, 0, :] = board_objects["wall"]
        board_values[:, -1, :] = board_objects["wall"]
        for i in range(1, len(self.XY) - 1):
            board_values[self.X[i], self.Y[i], :] = board_objects["body"]
        board_values[self.fx, self.fy, :] = board_objects["food"]
        board_values[self.X[-1], self.Y[-1], :] = board_objects["tail"]
        board_values[self.X[0], self.Y[0], :] = board_objects["head"]
        return board_values

    def next_move(self, direction):
        """
        Takes the desired direction as input and takes a single step and also
        returns the reward value and updated board values
        """
        self.direction = direction
        if direction == 0:  # move left
            new_X = self.X[0] - 1
            new_Y = self.Y[0]
        elif direction == 1:  # move up
            new_X = self.X[0]
            new_Y = self.Y[0] + 1
        elif direction == 2:  # move right
            new_X = self.X[0] + 1
            new_Y = self.Y[0]
        elif direction == 3:  # move down
            new_X = self.X[0]
            new_Y = self.Y[0] - 1
        else:
            self.game_over = 1
            raise ValueError("Invalid direction")
        new_XY = (new_X, new_Y)
        if (
            new_X > 0
            and new_Y > 0
            and new_X <= self.board_size
            and new_Y <= self.board_size
            and (new_XY not in self.XY[:-1])
        ):
            # valid moves - no snake body or wall in the new position
            # Note: The new head can be in the position of tail in the
            # previous step as this position would become free as snake moves
            self.X.insert(0, new_X)
            self.Y.insert(0, new_Y)
            self.XY.insert(0, new_XY)
            if new_X == self.fx and new_Y == self.fy:
                # Caught the food
                self.reward = rewards["food"]
                self.score = len(self.XY)
                if self.score == (self.board_size) ** 2:
                    # check if game is finished
                    self.game_over = 1
                else:
                    # Collect new food
                    self.fx, self.fy, self.food = self.get_new_food()
            else:
                # Clear the tail
                self.reward = rewards["none"]
                self.X.pop()
                self.Y.pop()
                self.XY.pop()
            self.board_values = self.get_board_vals()
        else:
            self.reward = rewards["hit"]
            self.game_over = 1
        return self.board_values, self.reward, self.game_over

    def draw_board(self, im=None):
        """
        Animates the game by displaying the board values as corresponding
        RGB pixel values
        """
        if im is None:
            plt.figure()
            ax = plt.axes()
            ax.axis("off")
            title = "Snake game "+str(self.board_size)+"x"+str(self.board_size)
            ax.set_title(title)
            im = plt.imshow(self.board_values)
        else:
            im.set_data(self.board_values)
        plt.pause(0.01)
        return im
