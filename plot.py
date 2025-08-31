import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_bankrolls_from_excel(filepath):
    # Load Excel
    df = pd.read_excel(filepath)

    # Remove the 'Agent' column and keep only numeric bankroll data
    data = df.drop(columns=['Agent'])
    labels = df['Agent'].tolist()

    # Organize runs and averages
    runs = {
        'DQN_untrained': [],
        'DQN_random': [],
        'DQN_self': [],
    }
    averages = {}

    for i, label in enumerate(labels):
        if 'avg' in label:
            if 'self' in label:
                averages['DQN_self'] = data.iloc[i].values
            elif 'random' in label:
                averages['DQN_random'] = data.iloc[i].values
            elif 'untrained' in label:
                averages['DQN_untrained'] = data.iloc[i].values
        else:
            if 'self' in label:
                runs['DQN_self'].append(data.iloc[i].values)
            elif 'random' in label:
                runs['DQN_random'].append(data.iloc[i].values)
            elif 'untrained' in label:
                runs['DQN_untrained'].append(data.iloc[i].values)

    # Plot

        # Set global font parameters
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    plt.figure(figsize=(12, 6))
    agent_styles = {
        'DQN_untrained': {'color': 'gray', 'label': 'DQN (untrained)'},
        'DQN_random': {'color': 'blue', 'label': 'DQN (trained against random)'},
        'DQN_self': {'color': 'green', 'label': 'DQN (self-trained)'},
    }

    for agent, runs_list in runs.items():
        for run in runs_list:
            plt.plot(run, alpha=0.2, color=agent_styles[agent]['color'])
        plt.plot(
            averages[agent],
            linewidth=2.5,
            label=agent_styles[agent]['label'],
            color=agent_styles[agent]['color']
        )

    # Axis labels and title
    plt.xlabel("Number of hands [-]", fontsize=12)
    plt.ylabel(r"$\Delta$ Bankroll [BB's]", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/bankroll_evolution.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_bankrolls_from_excel("results/evaluation_bankrolls.xlsx")
