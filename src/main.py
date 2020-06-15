import torch
import gym
from src.model import PPO, Memory
from src.problem import LinearRegression
from src.train import L2O, Baselines
import numpy as np
import argparse

def main(args):
    if args.method == 'PPO':
        L2O(args)
    else:
        Baselines(args)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default='LR', help="problems")
    parser.add_argument('--method', type=str, default='PPO', help="PPO, SGD, Momentum, ADAM, L-BFGS")
    args = parser.parse_args()



    main(args)