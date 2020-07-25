import numpy as np

from maze import BasicMaze
from dynamic_programming_solver import DPSolver
from utils import prep_results, prep_arrows
from q_learner import train as q_train
from sarsa_learner import train as sarsa_train
import argparse
import json


STAGES = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description='Maze solver via different RL algorithms')
    parser.add_argument('input', type=str, help='filepath to input json file')
    parser.add_argument('output', type=str,
                        help='filepath of json file to output')
    parser.add_argument('--max-iterations', metavar='max_its', type=int,
                        default=100, help='maximum number of iterations to be run')
    return parser.parse_args()


def solve(matrix, **kwargs):
    # Convert matrix into maze opject
    maze = BasicMaze(matrix)

    # Switch cases for different algorithms
    return DPSolver(maze, **kwargs)


def execute_dp(params, **kwargs):
    results = solve(**params, **kwargs)
    arrows = prep_arrows(results)
    return {"values": prep_results(results), "n": len(results), "arrows": arrows}


def execute_td_learning(params, policy, **kwargs):
    if policy == 'sarsa':
        trainer = sarsa_train
    elif policy == 'ql':
        trainer = q_train

    q_table_history, epsilon_history = trainer(**params)

    epsilon_history[-1] = 0.0

    dim = len(params['matrix'])
    history = []

    for q_table in q_table_history:
        result = []
        for i in range(dim):
            result_ = []
            for j in range(dim):
                result_.append(q_table[(i, j)].tolist())
            result.append(result_)
        history.append(result)

    sample_history = []
    sample_epsilon_history = []

    interval = int(len(history) / STAGES)

    if len(history) < STAGES:
        sample_history, epsilon_history = history, epsilon_history
    else:
        for i in range(0, len(history), interval):
            sample_history.append(history[i])
            sample_epsilon_history.append(epsilon_history[i])

    return {"history": sample_history, "n": len(sample_history), "epsilon": sample_epsilon_history}


if __name__ == '__main__':
    args = parse_args()
    with open(args.input, 'r') as f:
        params = json.load(f)

    with open(args.output, 'w') as f:
        results = solve(**params, **(vars(args)))
        results = prep_results(results)
        json.dump({"values": results,
                   "n": len(results)}, f)
