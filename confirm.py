import pandas as pd

df = pd.read_excel("results/sensitivity_curves.xlsx")

# Check how many rows exist for epsilon_decay_steps = 10000 and training_type = 'random'
subset_random = df[
    (df['varied_param'] == 'epsilon_decay_steps') &
    (df['varied_value'] == 10000) &
    (df['training_type'] == 'random')
]
print("Random agent, decay 10000 rows:", len(subset_random))

# Repeat for self-trained agent
subset_self = df[
    (df['varied_param'] == 'epsilon_decay_steps') &
    (df['varied_value'] == 10000) &
    (df['training_type'] == 'self')
]
print("Self agent, decay 10000 rows:", len(subset_self))

# Check for decay = 20000
subset_self_20000 = df[
    (df['varied_param'] == 'epsilon_decay_steps') &
    (df['varied_value'] == 20000) &
    (df['training_type'] == 'self')
]
print("Self agent, decay 20000 rows:", len(subset_self_20000))
