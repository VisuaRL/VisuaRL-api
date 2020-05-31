import numpy as np

from maze import BasicMaze
from utils import prep_val_est_arr, arr_print


# Non in-place version of Dynamic Programming algorithm
def DPSolver(maze, gamma=1, improvement=1e-5, max_iterations=10, **kwargs):
    # Initialize value array
    value_arr = prep_val_est_arr(maze.board)
    improved_value_arr = np.array(value_arr, copy=True)
    rows, cols = value_arr.shape

    value_estimate_history = []

    iterations = 0

    while True:
        iterations += 1
        for row in range(rows):
            for col in range(cols):
                # If coordinate is empty (ie not a valid space on the maze)
                if maze.is_empty(row, col):
                    continue

                # Get legal moves and get their state value
                actions = maze.get_available_moves(row, col)

                # Next state values and max value
                if not maze.is_goal_state(row, col):
                    state_values = []
                    max_value = np.iinfo(np.int64).min

                    # Get state values
                    for action in actions:
                        next_state = maze.get_next_state(row, col, action)
                        next_state_val = value_arr[next_state]
                        state_values.append((next_state, next_state_val))
                        if next_state_val > max_value:
                            max_value = next_state_val

                    improved_value_arr[row, col] = maze.get_state_reward(row, col) + gamma * max_value
                    # print(f'coord: {row}, {col} rew: {maze.get_state_reward(row, col)}, delta: {gamma * max_value} new_val: {maze.get_state_reward(row, col) + gamma * max_value}')

                else:

                    improved_value_arr[row, col] = maze.get_state_reward(row, col)

        value_estimate_history.append(np.array(improved_value_arr, copy=True))

        value_arr = np.array(improved_value_arr, copy=True)
        improved_value_arr = np.array(value_arr, copy=True)

        if len(value_estimate_history) == 1:
            continue
        else:
            max_error = np.abs(value_estimate_history[-1] - value_estimate_history[-2]).mean()
            if max_error < improvement or iterations > max_iterations:
                break

    return value_estimate_history[:-1]
