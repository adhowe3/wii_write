import pandas as pd

# Load your extracted features
df = pd.read_csv("extracted_features.csv")

# Group by the letter label and calculate the average for the means
summary = df.groupby('label')[['x_mean', 'y_mean', 'z_mean']].mean().round(2)

print("=== AVERAGE BASELINE EXPECTATIONS PER LETTER ===")
print(summary.to_string())
print("================================================")