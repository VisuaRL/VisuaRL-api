import numpy as np

from maze_env import MazeEnv

import copy


def run_episode(env, q_table, epsilon, gamma, alpha, max_ep_t=100, test=False):
    rewards = 0.0
    obs = env.reset()

    done = False

    prev_obs = None
    prev_act = None
    prev_rew = None

    for t in range(max_ep_t):
        # get available actions
        avail_actions = env.get_available_actions()

        # select action
        q_values = q_table[obs]
        mask = np.ones(q_values.shape, bool)
        mask[avail_actions] = 0
        q_values[mask] = -99999

        action = np.argmax(q_values)

        if test is False and np.random.random() < epsilon:
            action = np.random.choice(avail_actions)

        # execute action
        next_obs, rew, done = env.step(action)

        # update Q-value
        if prev_obs or prev_act or prev_rew:
            old_q = q_table[prev_obs][prev_act]
            next_q_value = q_table[obs][action]
            q_table[prev_obs][prev_act] = old_q + alpha * \
                (prev_rew + gamma * next_q_value - old_q)

        # upate obs
        prev_obs = obs
        prev_act = action
        prev_rew = rew
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
            q_table[(i, j)] = np.array([-50, -50, -50, -50], np.float)

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
