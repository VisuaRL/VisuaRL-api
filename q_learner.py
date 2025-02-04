import numpy as np

from maze_env import MazeEnv

import copy


def run_episode(env, q_table, epsilon, gamma, alpha, max_ep_t=100, test=False):
    rewards = 0.0
    obs = env.reset()

    done = False

    for t in range(max_ep_t):
        # get available actions
        avail_actions = env.get_available_actions()

        # select action
        q_values = q_table[obs]
        mask = np.ones(q_values.shape, bool)
        mask[avail_actions] = 0
        q_values[mask] = -9999

        action = np.argmax(q_values)

        if test is False and np.random.random() < epsilon:
            action = np.random.choice(avail_actions)
            assert action in avail_actions

        # execute action
        next_obs, rew, done = env.step(action)

        # update Q-value
        old_q = q_table[obs][action]
        next_q_values = q_table[next_obs]
        q_table[obs][action] = old_q + alpha * \
            (rew + gamma * np.max(next_q_values) - old_q)

        # print(q_table[obs][action])

        # upate obs
        obs = next_obs
        # add rewards
        rewards += rew

        if done:
            break

    return rewards


def train(matrix, epsilonDecay, max_episodes=2000, gamma=0.99, alpha=0.1, **kwargs):
    env = MazeEnv(matrix)

    q_table = {}

    rows, cols = env.maze.board.shape
    for i in range(rows):
        for j in range(cols):
            q_vals = np.array([-50, -50, -50, -50], np.float)
            if i - 1 >= 0:
                if env.maze.board[i-1, j] == np.iinfo(np.int64).min:
                    q_vals[0] = -9999
            if i + 1 < rows:
                if env.maze.board[i+1, j] == np.iinfo(np.int64).min:
                    q_vals[1] = -9999
            if j - 1 >= 0:
                if env.maze.board[i, j-1] == np.iinfo(np.int64).min:
                    q_vals[2] = -9999
            if j + 1 < cols:
                if env.maze.board[i, j+1] == np.iinfo(np.int64).min:
                    q_vals[3] = -9999
            q_table[(i, j)] = q_vals

    epsilon = 1.0

    rew_n = []

    q_table_history = []
    epsilon_history = []
    rew_history = []

    for episode in range(max_episodes):

        rew = run_episode(env, q_table, epsilon, gamma=gamma,
                          alpha=alpha, test=False)

        rew_n.append(rew)

        epsilon *= epsilonDecay

        if episode % 20 == 0:
            print(
                f'episode: {episode} average rewards: {np.average(rew_n[-100:])}')
            q_table_history.append(copy.deepcopy(q_table))
            epsilon_history.append(epsilon)
            rew_history.append(np.average(rew_n[-100:]))
            if abs(rew_history[-1] - np.average(rew_history[-4:-1])) < 1 and episode > 100:
                break

    return q_table_history, epsilon_history, rew_n


if __name__ == '__main__':
    import json
    with open('data/sample.json', 'r') as f:
        data = json.load(f)

        train(**data)
