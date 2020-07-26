import numpy as np

from utils import arr_print, prep_maze


class BasicMaze(object):

    def __init__(self, board):
        if not isinstance(board, np.ndarray):
            board = prep_maze(np.array(board))
        self.board = self.validate_board(board)
        self.rewards  = self.initialize()

    def validate_board(self, board):
        # Ensure that there is an actual valid path from the starting position to the goal square

        # Check that goal is indicated
        assert 3 in board

        # Check for available path from start to goal
        # TODO: Implement checker using BFS or some other algorithm

        return board

    def get_start(self):
        rows, cols = self.board.shape
        for i in range(rows):
            for j in range(cols):
                if self.board[i,j] == 2:
                    return (i,j)

    def initialize(self):
        # Initialize value function array
        rewards_arr = np.zeros_like(self.board)

        # Set appropriate values for rewards arr
        empty_cells = self.board == 0
        rewards_arr[empty_cells] = np.iinfo(np.int64).min
        rewards_arr[np.logical_not(empty_cells)] = -10
        rewards_arr[self.board == 3] = 10

        return rewards_arr

    def get_available_moves(self, x, y):
        # Ensure that queried cell is not an empty cell
        assert self.board[x,y] != np.iinfo(np.int64).min

        if self.is_goal_state(x, y):
            return []

        # Return available moves from a given state
        rows, cols = self.board.shape
        moves = []
        if x - 1 >= 0:
            if self.board[x-1,y] != np.iinfo(np.int64).min: moves.append('UP')
        if x + 1 < rows:
            if self.board[x+1,y] != np.iinfo(np.int64).min: moves.append('DOWN')
        if y - 1 >= 0:
            if self.board[x,y-1] != np.iinfo(np.int64).min: moves.append('LEFT')
        if y + 1 < cols:
            if self.board[x,y+1] != np.iinfo(np.int64).min: moves.append('RIGHT')
        return moves

    def is_empty(self, x, y):
        return self.board[x,y] == np.iinfo(np.int64).min

    def get_next_state(self, x, y, action):
        if action == 'UP':
            return x - 1, y
        elif action == 'DOWN':
            return x + 1, y
        elif action == 'LEFT':
            return x, y - 1
        else:
            return x, y + 1

    def is_goal_state(self, x, y):
        return self.board[x, y] == 3

    def get_state_reward(self, x, y):
        return self.rewards[x, y]

    def __repr__(self):
        # Print out maze shape to screen
        return arr_print(self.board)
