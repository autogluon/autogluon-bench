import pandas as pd
import numpy as np
from scipy.stats import sem  # Import the sem function for standard error of mean calculation

input_file = 'classification_regression.csv'
output_file = 'result_file.csv'

df = pd.read_csv(input_file)
grouped = df.groupby(['framework', 'task'])

results = []

# Iterate over each group
for (framework, task), group in grouped:
    results_data = group['result'].dropna()

    mean = results_data.mean()
    se = sem(results_data)
    se_196 = se * 1.96

    results.append({
        'Framework': framework,
        'Task': task,
        'Result': f"{mean:.3f}({se_196:.3f})"
    })

results_df = pd.DataFrame(results)

results_df.sort_values(by=['Framework', 'Task'], inplace=True)

results_df.to_csv(output_file, index=False)

print(f"Results have been saved to {output_file}")

