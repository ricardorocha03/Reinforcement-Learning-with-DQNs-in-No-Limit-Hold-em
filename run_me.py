import numpy as np
from rlcard.envs import make
from dqn_agent import DQNAgent
from rlcard.agents import RandomAgent
import torch
from rlcard.utils import reorganize
import random
import os
import pandas as pd
from utils import set_global_seed, evaluate

def train_dqn(env, episodes, opponent):
    agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[512, 512]
    )

    # Set opponents
    if opponent == 'self':
        env.set_agents([agent, agent])
    else:
        env.set_agents([agent, opponent])

    losses = []

    for episode in range(episodes):
        trajectories, payoffs = env.run(is_training=True)

        # Convert trajectory into (s, a, r, s’, done)
        trajectories = reorganize(trajectories, payoffs)

        for ts in trajectories[0]:  # Only update learning agent (index 0)
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
            if hasattr(agent, 'loss'):
                losses.append(agent.loss)
        else:
            losses.append(None)



        if (episode + 1) % 100 == 0:
            print(f" \n Episode {episode + 1}/{episodes}")

    return agent, losses

def main():
    seeds = [1119, 34862, 51802, 53400, 76398]
    num_training_episodes = 100000
    num_eval_games = 1000

    os.makedirs("results", exist_ok=True)

    rows = []
    labels = []

    # Collect per-agent-type data for averaging
    dqn_self_runs = []
    dqn_random_runs = []
    dqn_untrained_runs = []

    for seed in seeds:
        print(f"\n==== SEED {seed} ====")
        set_global_seed(seed)

        env_self_play = make('no-limit-holdem', config={'seed': seed})
        env_vs_random = make('no-limit-holdem', config={'seed': seed})
        env_baseline = make('no-limit-holdem', config={'seed': seed})

        random_agent = RandomAgent(env_vs_random.num_actions)

        # === DQN vs Self ===
        print("Training DQN vs DQN...")
        dqn_vs_self, _ = train_dqn(env_self_play, num_training_episodes, opponent='self')
        env_self_play.set_agents([dqn_vs_self, dqn_vs_self])
        bankroll_self = evaluate(env_self_play, dqn_vs_self, random_agent, num_eval_games)
        dqn_self_runs.append(bankroll_self)
        rows.append(bankroll_self)
        labels.append(f'DQN_self_seed{seed}')

        # === DQN vs Random ===
        print("Training DQN vs Random...")
        dqn_vs_random, _ = train_dqn(env_vs_random, num_training_episodes, opponent=random_agent)
        bankroll_random = evaluate(env_vs_random, dqn_vs_random, random_agent, num_eval_games)
        dqn_random_runs.append(bankroll_random)
        rows.append(bankroll_random)
        labels.append(f'DQN_random_seed{seed}')

        # === Untrained DQN ===
        print("Evaluating Untrained DQN...")
        dqn_untrained = DQNAgent(
            num_actions=env_baseline.num_actions,
            state_shape=env_baseline.state_shape[0],
            mlp_layers=[512, 512]
        )
        bankroll_untrained = evaluate(env_baseline, dqn_untrained, random_agent, num_eval_games)
        dqn_untrained_runs.append(bankroll_untrained)
        rows.append(bankroll_untrained)
        labels.append(f'DQN_untrained_seed{seed}')

    # === AVERAGES ===
    def average_rows(runs):
        return list(np.mean(runs, axis=0))

    rows.append(average_rows(dqn_self_runs))
    labels.append('DQN_self_avg')

    rows.append(average_rows(dqn_random_runs))
    labels.append('DQN_random_avg')

    rows.append(average_rows(dqn_untrained_runs))
    labels.append('DQN_untrained_avg')

    # === SAVE ===
    df = pd.DataFrame(rows)
    df.insert(0, "Agent", labels)
    df.to_excel("results/evaluation_bankrolls.xlsx", index=False)

if __name__ == '__main__':
    main()