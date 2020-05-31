import numpy as np

from maze import BasicMaze
from dynamic_programming_solver import DPSolver
from utils import prep_results

import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Maze solver via different RL algorithms')
    parser.add_argument('input', type=str, help='filepath to input json file')
    parser.add_argument('output', type=str, help='filepath of json file to output')
    parser.add_argument('--max-iterations', metavar='max_its', type=int, default=100, help='maximum number of iterations to be run')
    return parser.parse_args()


def solve(matrix, algo, **kwargs):
    # Convert matrix into maze opject
    maze = BasicMaze(matrix)

    # Switch cases for different algorithms
    if algo == 'dp':
        return DPSolver(maze, **kwargs)

    else:
        raise Exception(f'Algorithm {algo} not recognized')


def execute_solver(input_fp, output_fp, **kwargs):
    with open(input_fp, 'r') as f:
        params = json.load(f)

    with open(output_fp, 'w') as f:
        results = solve(**params, **kwargs)
        results = prep_results(results)
        json.dump({"stateValues": results,
                   "n":len(results)}, f)


if __name__ == '__main__':
    args = parse_args()
    with open(args.input, 'r') as f:
        params = json.load(f)

    with open(args.output, 'w') as f:
        results = solve(**params, **(vars(args)))
        results = prep_results(results)
        json.dump({"stateValues": results,
                   "n":len(results)}, f)
