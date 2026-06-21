import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os

# Create output directory if it doesn't exist
output_dir = 'experiment_results_two'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load result files
print("Loading result files...")
df_5users = pd.read_csv('experiments/fyp_best-5users-10runs-rand/results.csv')
df_10users = pd.read_csv('experiments/fyp_best-10users-10runs-rand/results.csv')
df_15users = pd.read_csv('experiments/fyp_best-15users-10runs-rand/results.csv')
df_20users = pd.read_csv('experiments/fyp_best-20users-10runs-rand/results.csv')

print(f"Data loaded successfully. Found {len(df_5users)} runs for 5 users, {len(df_10users)} runs for 10 users, " 
      f"{len(df_15users)} runs for 15 users, and {len(df_20users)} runs for 20 users.")

# Add user count column to each dataframe
df_5users['user_count'] = 5
df_10users['user_count'] = 10
df_15users['user_count'] = 15
df_20users['user_count'] = 20

# Combine all data
combined_df = pd.concat([df_5users, df_10users, df_15users, df_20users])

# Generate summary statistics
print("\nGenerating summary statistics...")
summary = combined_df.groupby('user_count')['accuracy'].agg(['mean', 'std', 'min', 'max'])
print("\nAccuracy Summary Statistics:")
print(summary)

# Calculate training time statistics
time_summary = combined_df.groupby('user_count')['training_time'].agg(['mean', 'std', 'min', 'max'])
print("\nTraining Time Summary Statistics (seconds):")
print(time_summary)

# Identify gesture columns
gesture_cols = [col for col in combined_df.columns if col.startswith('gesture_') and col.endswith('_accuracy')]
gesture_names = [f'Gesture {col.split("_")[1]}' for col in gesture_cols]

# Calculate means for each gesture and user count
print("\nCalculating gesture-specific statistics...")
gesture_stats = {}
for users, df in [(5, df_5users), (10, df_10users), (15, df_15users), (20, df_20users)]:
    gesture_means = {}
    for col in gesture_cols:
        # Skip NaN values
        gesture_means[col] = df[col].dropna().mean()
    gesture_stats[users] = gesture_means

# Print gesture-specific mean accuracies
print("\nGesture-Specific Mean Accuracies:")
for users in [5, 10, 15, 20]:
    print(f"\n{users} Users:")
    for col, name in zip(gesture_cols, gesture_names):
        if col in gesture_stats[users]:
            print(f"  {name}: {gesture_stats[users][col]:.4f}")

# VISUALIZATION 1: Overall Accuracy by User Count
print("\nGenerating accuracy boxplot...")
plt.figure(figsize=(10, 6))
sns.boxplot(x='user_count', y='accuracy', data=combined_df)
plt.title('EMG-Based User Identification Accuracy vs. Number of Users', fontsize=14)
plt.xlabel('Number of Users', fontsize=12)
plt.ylabel('Identification Accuracy', fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Add mean values as formatted x-axis labels
user_counts = [5, 10, 15, 20]
xlabels = []
for users in user_counts:
    mean_acc = combined_df[combined_df['user_count'] == users]['accuracy'].mean()
    xlabels.append(f"{users}\n({mean_acc:.3f})")

plt.xticks(range(len(user_counts)), xlabels)

# Adjust y-axis limits to focus on the relevant range
plt.ylim(0.83, 1.01)

plt.tight_layout()
plt.savefig(f'{output_dir}/accuracy_vs_users.png', dpi=300)
plt.close()
print(f"Saved to {output_dir}/accuracy_vs_users.png")

# VISUALIZATION 2: Per-Gesture Accuracy Analysis
print("\nGenerating gesture-specific accuracy comparison...")
plt.figure(figsize=(14, 8))

# Replace generic gesture names with actual gesture names, excluding gesture 1
gesture_names = [
    "Hand Clenched", 
    "Wrist Flexion",
    "Wrist Extension",
    "Radial Deviations",
    "Ulnar Deviations",
    "Extended Palm"
]

# Extract means for each gesture, handling potential NaN values
means_5 = [df_5users[col].dropna().mean() if not df_5users[col].dropna().empty else 0 for col in gesture_cols]
means_10 = [df_10users[col].dropna().mean() if not df_10users[col].dropna().empty else 0 for col in gesture_cols]
means_15 = [df_15users[col].dropna().mean() if not df_15users[col].dropna().empty else 0 for col in gesture_cols]
means_20 = [df_20users[col].dropna().mean() if not df_20users[col].dropna().empty else 0 for col in gesture_cols]

# Set width of bars
barWidth = 0.2
r1 = np.arange(len(gesture_names))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

# Create grayscale colors (light to dark)
colors = ['#D3D3D3', '#A9A9A9', '#696969', '#000000']  # Light gray to black

# Create grouped bar chart with grayscale colors
plt.bar(r1, means_5, width=barWidth, label='5 Users', color=colors[0])
plt.bar(r2, means_10, width=barWidth, label='10 Users', color=colors[1])
plt.bar(r3, means_15, width=barWidth, label='15 Users', color=colors[2])
plt.bar(r4, means_20, width=barWidth, label='20 Users', color=colors[3])

# Add labels and legend
plt.xlabel('Gesture Type', fontsize=12)
plt.ylabel('Average Accuracy', fontsize=12)
plt.title('Gesture-Specific Accuracy Across Different User Counts', fontsize=14)
plt.xticks([r + barWidth*1.5 for r in range(len(gesture_names))], gesture_names, rotation=15)
plt.legend()

# Add grid for better readability
plt.grid(axis='y', alpha=0.3)
plt.ylim(0.5, 1.0)  # Focus on the relevant accuracy range

plt.tight_layout()
plt.savefig(f'{output_dir}/gesture_accuracy_comparison.png', dpi=300)
plt.close()
print(f"Saved to {output_dir}/gesture_accuracy_comparison.png")

# VISUALIZATION 2b: Grouped Line Plot for Gesture Accuracies
print("\nGenerating gesture-specific accuracy line plot...")
plt.figure(figsize=(12, 7))

# Create line plot data
user_counts = [5, 10, 15, 20]
for i, gesture in enumerate(gesture_names):
    accuracies = [
        df_5users[gesture_cols[i]].dropna().mean() if not df_5users[gesture_cols[i]].dropna().empty else 0,
        df_10users[gesture_cols[i]].dropna().mean() if not df_10users[gesture_cols[i]].dropna().empty else 0,
        df_15users[gesture_cols[i]].dropna().mean() if not df_15users[gesture_cols[i]].dropna().empty else 0,
        df_20users[gesture_cols[i]].dropna().mean() if not df_20users[gesture_cols[i]].dropna().empty else 0
    ]
    plt.plot(user_counts, accuracies, 'o-', linewidth=2, markersize=8, label=gesture)

# Add labels and legend
plt.xlabel('Number of Users', fontsize=12)
plt.ylabel('Average Accuracy', fontsize=12)
plt.title('Gesture-Specific Accuracy Trends Across User Counts', fontsize=14)
plt.xticks(user_counts)
plt.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), 
           mode="expand", ncol=3, fontsize=10)

# Add grid for better readability
plt.grid(alpha=0.3)
plt.ylim(0.5, 1.0)  # Focus on the relevant accuracy range

plt.tight_layout()
plt.savefig(f'{output_dir}/gesture_accuracy_lineplot.png', dpi=300)
plt.close()
print(f"Saved to {output_dir}/gesture_accuracy_lineplot.png")

# VISUALIZATION 2c: Heatmap for Gesture Accuracies
print("\nGenerating gesture-specific accuracy heatmap...")
plt.figure(figsize=(10, 7))

# Create data for heatmap
heatmap_data = np.zeros((len(gesture_names), len(user_counts)))
for i, gesture in enumerate(gesture_names):
    heatmap_data[i, 0] = df_5users[gesture_cols[i]].dropna().mean() if not df_5users[gesture_cols[i]].dropna().empty else 0
    heatmap_data[i, 1] = df_10users[gesture_cols[i]].dropna().mean() if not df_10users[gesture_cols[i]].dropna().empty else 0
    heatmap_data[i, 2] = df_15users[gesture_cols[i]].dropna().mean() if not df_15users[gesture_cols[i]].dropna().empty else 0
    heatmap_data[i, 3] = df_20users[gesture_cols[i]].dropna().mean() if not df_20users[gesture_cols[i]].dropna().empty else 0

# Create heatmap with a blue color palette (lighter to darker)
ax = sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='Blues',
            xticklabels=user_counts, yticklabels=gesture_names,
            vmin=0.5, vmax=1.0, cbar_kws={'label': 'Accuracy'})

# Customize heatmap
plt.title('Gesture-Specific Accuracy Across User Counts', fontsize=14)
plt.xlabel('Number of Users', fontsize=12)
plt.ylabel('Gesture Type', fontsize=12)

plt.tight_layout()
plt.savefig(f'{output_dir}/gesture_accuracy_heatmap.png', dpi=300)
plt.close()
print(f"Saved to {output_dir}/gesture_accuracy_heatmap.png")

# VISUALIZATION 2d: Radar Chart (Spider Chart) for Gesture Accuracies
print("\nGenerating gesture-specific accuracy radar chart...")
plt.figure(figsize=(10, 9))

# Set data
angles = np.linspace(0, 2*np.pi, len(gesture_names), endpoint=False).tolist()
angles += angles[:1]  # Close the loop

# Make copies of the means arrays for the radar chart and close the loop
radar_means_5 = means_5.copy()
radar_means_10 = means_10.copy()
radar_means_15 = means_15.copy()
radar_means_20 = means_20.copy()
radar_means_5 += [radar_means_5[0]]
radar_means_10 += [radar_means_10[0]]
radar_means_15 += [radar_means_15[0]]
radar_means_20 += [radar_means_20[0]]

# Set up the radar chart
ax = plt.subplot(111, polar=True)

# Define the requested color scheme
colors = ['#2ca02c', '#a1a102', '#ff8c00', '#d62728']  # Green, Yellow, Dark Orange, Red

# Plot each user count data with the specified colors - no fill, just lines
ax.plot(angles, radar_means_5, 'o-', linewidth=2.5, label='5 Users', color=colors[0])
ax.plot(angles, radar_means_10, 'o-', linewidth=2.5, label='10 Users', color=colors[1])
ax.plot(angles, radar_means_15, 'o-', linewidth=2.5, label='15 Users', color=colors[2])
ax.plot(angles, radar_means_20, 'o-', linewidth=2.5, label='20 Users', color=colors[3])

# Removed fill commands to show only the outlines

# Hide the gesture labels as requested
labels = gesture_names + [gesture_names[0]]  # Create the labels list but won't display them
plt.xticks(angles, [''] * len(labels), size=11)

# Set chart properties
ax.set_ylim(0.7, 1.0)  # Focus on the relevant accuracy range
ax.grid(True, alpha=0.3)

# Draw y-axis ticks and labels
ax.set_yticks([0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
ax.set_yticklabels(['0.75', '0.8', '0.85', '0.9', '0.95', '1.0'], fontsize=9)

# Add legend with a better position
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=True, fontsize=10)

plt.title('Gesture-Specific Accuracy Across User Counts', fontsize=14)
plt.tight_layout()
plt.savefig(f'{output_dir}/gesture_accuracy_radar.png', dpi=300)
plt.close()
print(f"Saved to {output_dir}/gesture_accuracy_radar.png")

# VISUALIZATION 3: Computational Performance Analysis
print("\nGenerating training time boxplot...")
plt.figure(figsize=(10, 6))
sns.boxplot(x='user_count', y='training_time', data=combined_df)
plt.title('Training Time vs. Number of Users', fontsize=14)
plt.xlabel('Number of Users', fontsize=12)
plt.ylabel('Training Time (seconds)', fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Add text annotations with mean values
for i, users in enumerate([5, 10, 15, 20]):
    mean_time = combined_df[combined_df['user_count'] == users]['training_time'].mean()
    plt.text(i, combined_df['training_time'].min() - 1, 
             f'Mean: {mean_time:.1f}s', 
             horizontalalignment='center')

plt.tight_layout()
plt.savefig(f'{output_dir}/training_time_vs_users.png', dpi=300)
plt.close()
print(f"Saved to {output_dir}/training_time_vs_users.png")

# VISUALIZATION 4: Accuracy-Time Trade-off
print("\nGenerating accuracy-time trade-off plot...")
plt.figure(figsize=(9, 7))

# Calculate mean values for each user count
user_counts = [5, 10, 15, 20]
acc_means = []
time_means = []

for users in user_counts:
    df = combined_df[combined_df['user_count'] == users]
    acc_means.append(df['accuracy'].mean())
    time_means.append(df['training_time'].mean())

# Create scatter plot with connecting line
plt.plot(time_means, acc_means, 'o-', markersize=10)

# Add annotations for each point
for i, users in enumerate(user_counts):
    plt.annotate(f'{users} Users', 
                 (time_means[i], acc_means[i]),
                 xytext=(10, -10),
                 textcoords='offset points',
                 fontsize=12)

plt.title('Accuracy-Time Trade-off with Increasing User Count', fontsize=14)
plt.xlabel('Average Training Time (seconds)', fontsize=12)
plt.ylabel('Average Identification Accuracy', fontsize=12)
plt.grid(alpha=0.3)
plt.ylim(0.85, 1.0)  # Focus on the relevant accuracy range

plt.tight_layout()
plt.savefig(f'{output_dir}/accuracy_time_tradeoff.png', dpi=300)
plt.close()
print(f"Saved to {output_dir}/accuracy_time_tradeoff.png")

# STATISTICAL ANALYSIS
print("\nPerforming statistical analysis...")

# Get accuracy values for each user count
acc_5 = df_5users['accuracy']
acc_10 = df_10users['accuracy']
acc_15 = df_15users['accuracy']
acc_20 = df_20users['accuracy']

# Perform one-way ANOVA
f_val, p_val = stats.f_oneway(acc_5, acc_10, acc_15, acc_20)
print(f"\nANOVA results for accuracy differences: F={f_val:.4f}, p={p_val:.4f}")

# If p < 0.05, perform post-hoc Tukey's test
if p_val < 0.05:
    tukey = pairwise_tukeyhsd(endog=combined_df['accuracy'], 
                             groups=combined_df['user_count'], 
                             alpha=0.05)
    print("\nTukey's HSD post-hoc test:")
    print(tukey)
    
    # Save Tukey results to file
    with open(f'{output_dir}/tukey_results.txt', 'w') as f:
        f.write(str(tukey))
    print(f"Saved Tukey results to {output_dir}/tukey_results.txt")
    
    # Create visualization of Tukey results
    plt.figure(figsize=(10, 6))
    
    # Get the pairs, mean differences, and p-values directly from tukey results
    group_pairs = [f"{pair[0]}-{pair[1]}" for pair in tukey._multicomp.pairindices]
    mean_diffs = tukey.meandiffs
    
    # Create bar chart of mean differences
    plt.bar(group_pairs, mean_diffs)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Mean Differences Between User Counts (Tukey HSD)', fontsize=14)
    plt.ylabel('Mean Difference in Accuracy', fontsize=12)
    plt.xlabel('Group Pairs', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tukey_plot.png', dpi=300)
    plt.close()
    print(f"Saved Tukey plot to {output_dir}/tukey_plot.png")

# Generate CSV summary of all results
print("\nGenerating summary CSV file...")
summary_dict = {
    'User Count': [],
    'Mean Accuracy': [],
    'Std Dev Accuracy': [],
    'Mean Training Time (s)': [],
    'Std Dev Training Time (s)': []
}

for users in [5, 10, 15, 20]:
    df = combined_df[combined_df['user_count'] == users]
    summary_dict['User Count'].append(users)
    summary_dict['Mean Accuracy'].append(df['accuracy'].mean())
    summary_dict['Std Dev Accuracy'].append(df['accuracy'].std())
    summary_dict['Mean Training Time (s)'].append(df['training_time'].mean())
    summary_dict['Std Dev Training Time (s)'].append(df['training_time'].std())

summary_df = pd.DataFrame(summary_dict)
summary_df.to_csv(f'{output_dir}/experiment_summary.csv', index=False)
print(f"Saved summary to {output_dir}/experiment_summary.csv")

print("\nAnalysis complete! All visualizations and statistics have been generated.")