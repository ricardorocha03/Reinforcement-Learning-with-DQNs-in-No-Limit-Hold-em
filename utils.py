# utils_rl.py

import random
import numpy as np
import torch

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(env, agent0, agent1, num_games):
    env.set_agents([agent0, agent1])
    bankrolls = [0]
    balance = 0

    for _ in range(num_games):
        _, payoffs = env.run(is_training=False)
        balance += payoffs[0]
        bankrolls.append(balance)

    return bankrolls