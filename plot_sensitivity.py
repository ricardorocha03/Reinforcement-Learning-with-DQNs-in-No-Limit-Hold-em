import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_sensitivity_from_excel(sensitivity_path, baseline_path):
    # Define baseline parameter values
    BASELINES = {
        "learning_rate": 5e-5,
        "discount_factor": 0.99,
        "epsilon_start": 1.0,
    }

    # Latex-friendly labels for plotting
    GREEK_MAP = {
        "learning_rate": r"$\alpha$",
        "discount_factor": r"$\gamma$",
        "epsilon_start": r"$\epsilon_{start}$",
    }

    # Load sensitivity data
    df = pd.read_excel(sensitivity_path)
    df['varied_value_str'] = df['varied_value'].apply(lambda x: f"{x:.0e}" if x < 0.001 else f"{x:.5g}")

    # Load baseline data
    df_baseline_raw = pd.read_excel(baseline_path)
    baseline_rows = []

    for param in BASELINES:
        baseline_val = BASELINES[param]
        val_str = f"{baseline_val:.0e}" if baseline_val < 0.001 else f"{baseline_val:.5g}"

        for i, label in enumerate(df_baseline_raw['Agent']):
            if 'avg' not in label:
                continue
            agent_type = 'random' if 'random' in label else 'self' if 'self' in label else None
            if agent_type is None:
                continue

            row = df_baseline_raw.iloc[i, 1:].reset_index(drop=True).to_frame(name='bankroll')
            row['hand'] = row.index
            row['varied_param'] = param
            row['varied_value'] = baseline_val
            row['varied_value_str'] = val_str
            row['training_type'] = agent_type
            row['seed'] = -1  # baseline
            baseline_rows.append(row)

    df_baseline = pd.concat(baseline_rows, ignore_index=True)
    df_combined = pd.concat([df, df_baseline], ignore_index=True)

    # Set global font parameters
    plt.rcParams.update({
        "font.size": 22,
        "axes.labelsize": 24,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 22,
    })

    output_dir = "results/sensitivity_plots"
    os.makedirs(output_dir, exist_ok=True)

    for param in BASELINES:
        df_param = df_combined[df_combined['varied_param'] == param]
        all_value_strs = sorted(df_param['varied_value_str'].unique(), key=lambda x: float(x))
        palette = sns.color_palette("tab10", len(all_value_strs))
        value_color_map = {val_str: palette[i] for i, val_str in enumerate(all_value_strs)}

        for agent in ['self', 'random']:
            df_agent = df_param[df_param['training_type'] == agent]

            plt.figure(figsize=(12, 6))

            for val_str in all_value_strs:
                df_val = df_agent[df_agent['varied_value_str'] == val_str]
                seeds = df_val['seed'].unique()
                color = value_color_map[val_str]

                # Transparent individual runs
                for seed in seeds:
                    if seed == -1:
                        continue
                    df_seed = df_val[df_val['seed'] == seed]
                    if not df_seed.empty:
                        plt.plot(df_seed['hand'], df_seed['bankroll'],
                                 alpha=0.25, linewidth=1, color=color)

                # Solid average curve
                df_avg = df_val.groupby('hand')['bankroll'].mean().reset_index()
                if not df_avg.empty:
                    linestyle = '--' if val_str == f"{BASELINES[param]:.0e}" or val_str == f"{BASELINES[param]:.5g}" else '-'
                    plt.plot(df_avg['hand'], df_avg['bankroll'],
                             label=f"{GREEK_MAP[param]} = {val_str}",
                             linewidth=2.5, color=color, linestyle=linestyle)

            plt.xlabel("Number of hands [-]")
            plt.ylabel(r"$\Delta$ Bankroll [BB's]")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            save_path = os.path.join(output_dir, f"{param}_{agent}.png")
            plt.savefig(save_path, dpi=300)
            plt.close()

if __name__ == "__main__":
    plot_sensitivity_from_excel(
        sensitivity_path="results/sensitivity_curves.xlsx",
        baseline_path="results/evaluation_bankrolls.xlsx"
    )
