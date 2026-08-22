import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Create directory for plots
os.makedirs('plots', exist_ok=True)

# Load data
df = pd.read_csv('results_chunk512_999questions.csv')

# Set aesthetic parameters
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# 1. Performance vs Top_K for different Models (Method = 'default', RRF_K =20)
# We filter by a fixed RRF_K so that we don't have duplicated points for non-RRF models
df_fixed_rrf = df[df['RRF_K'] == 20].copy()
df_default = df_fixed_rrf[df_fixed_rrf['Method'] == 'default']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metrics = ['Recall', 'NDCG', 'MRR']
for i, metric in enumerate(metrics):
    sns.lineplot(data=df_default, x='Top_K', y=metric, hue='Model', marker='o', ax=axes[i])
    axes[i].set_title(f'{metric} vs Top_K (Method: default, RRF_K: 20)')
    axes[i].set_xlabel('Top_K')
    axes[i].set_ylabel(metric)

plt.tight_layout()
plt.savefig('plots/performance_vs_top_k.png')
plt.close()
print("Generated plots/performance_vs_top_k.png")

# 2. Impact of Method ('default' vs 'with_titles')
# Let's compare default vs with_titles for a specific Top_K = 80 (or 10 if 80 is not present), RRF_K = 60
# Let's find a Top_K that exists. Based on preview, 10 and 80 exist. We will use 10.
df_method_compare = df[(df['Top_K'] == 80) & (df['RRF_K'] == 20)]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, metric in enumerate(metrics):
    sns.barplot(data=df_method_compare, x='Model', y=metric, hue='Method', ax=axes[i])
    axes[i].set_title(f'{metric}: Default vs With Titles (Top_K=80)')
    axes[i].set_xlabel('Model')
    axes[i].set_ylabel(metric)
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('plots/method_comparison.png')
plt.close()
print("Generated plots/method_comparison.png")

# 3. Impact of RRF_K on RRF model performance
# Let's fix Top_K = 10 to clearly show the impact for a specific Top_K
df_rrf = df[(df['Model'] == 'rrf') & (df['Method'] == 'default') & (df['Top_K'] == 80)].copy()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, metric in enumerate(metrics):
    sns.lineplot(data=df_rrf, x='RRF_K', y=metric, marker='o', color='teal', ax=axes[i])
    axes[i].set_title(f'{metric} vs RRF_K (RRF Model, Top_K: 80)')
    axes[i].set_xlabel('RRF_K')
    axes[i].set_ylabel(metric)

plt.tight_layout()
plt.savefig('plots/rrf_k_impact.png')
plt.close()
print("Generated plots/rrf_k_impact.png")

# 4. Time Taken Comparison
# Let's see average time taken by Method
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=df, x='Method', y='Time_Taken_Sec', errorbar=None)
ax.set_title('Average Time Taken by Method')
ax.set_xlabel('Method')
ax.set_ylabel('Time Taken (Seconds)')

plt.tight_layout()
plt.savefig('plots/time_taken_method.png')
plt.close()
print("Generated plots/time_taken_method.png")

# 5. Time Taken by Retrieval Technique (Model) per Question
# Calculate average compute time per question (divided by 999)
time_cols = {
    'dense': 'Time_Dense',
    'sparse': 'Time_Sparse',
    'rrf': 'Time_RRF',
    'colbert': 'Time_Colbert',
    'cross_encoder': 'Time_CrossEncoder'
}

records = []
for method in df['Method'].unique():
    df_method = df[df['Method'] == method]
    for model, col in time_cols.items():
        if col in df.columns:
            # average over all configs for this method, then divide by 999 questions
            avg_time = df_method[col].mean() / 999.0
            records.append({'Method': method, 'Model': model, 'Time_Per_Question_Sec': avg_time})

df_time = pd.DataFrame(records)

fig, ax = plt.subplots(figsize=(10, 6))
# Compare Time_Per_Question_Sec for each Model, grouped by Method
sns.barplot(data=df_time, x='Model', y='Time_Per_Question_Sec', hue='Method', ax=ax)
ax.set_title('Average Compute Time per Question by Retrieval Technique (Log Scale)')
ax.set_xlabel('Retrieval Technique (Model)')
ax.set_ylabel('Time per Question (Seconds)')
ax.set_yscale('log')
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('plots/time_taken_model.png')
plt.close()
print("Generated plots/time_taken_model.png")

print("All plots have been successfully generated in the 'plots' directory.")

# Export metrics to JSON
metrics_summary = {
    "performance_vs_top_k": df_default.to_dict(orient="records"),
    "method_comparison": df_method_compare.to_dict(orient="records"),
    "rrf_k_impact": df_rrf.to_dict(orient="records"),
    "time_taken_per_question": df_time.to_dict(orient="records")
}

with open("metrics_summary.json", "w", encoding="utf-8") as f:
    json.dump(metrics_summary, f, indent=4)

print("Generated metrics_summary.json")
