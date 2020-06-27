from maze import BasicMaze

import numpy as np


move_to_idx = {
    'UP':0,
    'DOWN':1,
    'LEFT':2,
    'RIGHT':3
}

idx_to_move = {
    0: 'UP',
    1: 'DOWN',
    2: 'LEFT',
    3: 'RIGHT'
}

class MazeEnv(object):
    def __init__(self, board):
        self.board = board
        self.initialize()

    def initialize(self):
        self.maze = BasicMaze(self.board)
        self.state = self.maze.get_start()

    def reset(self):
        self.initialize()
        return self.state

    def step(self, action):
        # TODO: Check that move is available
        action = idx_to_move[action]
        next_state = self.maze.get_next_state(*self.state, action)
        rew = self.maze.get_state_reward(*self.state)
        done = self.maze.is_goal_state(*next_state)
        self.state = next_state
        return self.state, rew, done

    def get_available_actions(self):
        result = []
        moves = self.maze.get_available_moves(*self.state)
        for move in moves:
            result.append(move_to_idx[move])
        return np.array(result)

    def __repr__(self):
        return self.maze.__repr__()
