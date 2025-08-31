import os
import numpy as np
import pandas as pd
from rlcard.envs import make
from dqn_agent import DQNAgent
from rlcard.agents import RandomAgent
import torch
import random
from utils import set_global_seed, evaluate

def train_dqn_custom(env, episodes, opponent, config):
    agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[512, 512],
        learning_rate=config.get('learning_rate', 5e-5),
        discount_factor=config.get('discount_factor', 0.99),
        epsilon_start=config.get('epsilon_start', 1.0),
        epsilon_end=config.get('epsilon_end', 0.1),
        epsilon_decay_steps=config.get('epsilon_decay_steps', 20000),
    )

    # Set agents
    if opponent == 'self':
        env.set_agents([agent, agent])
    else:
        env.set_agents([agent, opponent])

    for _ in range(episodes):
        trajectories, payoffs = env.run(is_training=True)
        from rlcard.utils import reorganize
        trajectories = reorganize(trajectories, payoffs)

        for ts in trajectories[0]:
            state, action, reward, next_state, done = ts
            if not isinstance(state, dict) or not isinstance(next_state, dict):
                continue
            agent.feed_memory(
                state['obs'],
                action,
                reward,
                next_state['obs'],
                list(next_state['legal_actions'].keys()),
                done
            )

        if len(agent.memory.memory) >= agent.batch_size:
            agent.train()

    return agent


def run_sensitivity_analysis_with_curves(planned_runs, seeds, output_file='results/sensitivity_curves.xlsx'):
    os.makedirs("results", exist_ok=True)
    all_results = []

    for config in planned_runs:
        for seed in seeds:
            print(f"\n=== Param: {config['varied_param']} = {config['varied_value']}, Type: {config['training_type']}, Seed: {seed} ===")
            set_global_seed(seed)

            env = make('no-limit-holdem', config={'seed': seed})
            opponent = RandomAgent(env.num_actions) if config['training_type'] == 'random' else 'self'
            agent = train_dqn_custom(env, 100000, opponent, config)

            random_eval_agent = RandomAgent(env.num_actions)
            bankroll_curve = evaluate(env, agent, random_eval_agent, 1000)

            result = pd.DataFrame({
                'hand': list(range(len(bankroll_curve))),
                'bankroll': bankroll_curve,
                'varied_param': config['varied_param'],
                'varied_value': config['varied_value'],
                'training_type': config['training_type'],
                'seed': seed
            })
            all_results.append(result)

    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_excel(output_file, index=False)
    print(f"\n✅ Saved to {output_file}")


if __name__ == '__main__':
    seeds = [1119, 34862, 51802, 53400, 76398]

    baseline = {
        'learning_rate': 5e-5,
        'discount_factor': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.1,
        'epsilon_decay_steps': 20000,
    }

    param_grid = {
        'learning_rate': [1e-5, 1e-4],
        'discount_factor': [0.95, 1.0],
        'epsilon_start': [0.7, 0.9],
        'epsilon_end': [0.01, 0.2],
        'epsilon_decay_steps': [10000, 40000],
    }

    planned_runs = []
    for param, values in param_grid.items():
        for val in values:
            for train_type in ['self', 'random']:
                config = baseline.copy()
                config[param] = val
                config['varied_param'] = param
                config['varied_value'] = val
                config['training_type'] = train_type
                planned_runs.append(config)

    run_sensitivity_analysis_with_curves(planned_runs, seeds)