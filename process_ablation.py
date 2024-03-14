import pandas as pd

# Load the processed data
df = pd.read_csv('classification_regression.csv')

# Define the desired dataset (Task) order
task_order = [
    "fasion_mnist", "food101", "stanfordcars", "magnetictiledefects",
    "europeanflooddepth", "oxfordflowers", "OxfordIIITPet", "cd18", "ham10000",
    "hateful_meme", "petfinder", "memotion", "financial_news", "MLDoc-11000",
    "MultiATIS-5000", "fb_dialog", "SNIPS", "ag_news", "airbnb", "kick_start",
    "cloth_review", "news_popularity", "cal_house"
]

# Pivot the DataFrame
pivoted_df = df.pivot(index='task', columns='framework', values='result')

# Ensure the DataFrame rows follow the specified task order
# Reindex the DataFrame according to the task_order list, this will automatically sort the rows
pivoted_df = pivoted_df.reindex(task_order)

# Specify the desired column (Framework) order
column_order = [
    'autokeras_master',
    "ablation_base", 
    "ablation_greedy_soup", 
    "ablation_gradient_clip", 
    "ablation_warmup_steps", 
    "ablation_cosine_decay", 
    "ablation_weight_decay", 
    "ablation_lr_decay"
]

# Reorder the columns according to the specified order
pivoted_df = pivoted_df[column_order]

# Save the reformatted DataFrame to a new CSV file
pivoted_df.to_csv('reformatted_results.csv')

print("Reformatted results saved to 'reformatted_results.csv'.")

