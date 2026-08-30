import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\batch_correlation.csv')

plt.figure(figsize=(10, 6))

# Group data by sample for boxplot-like plotting using pure matplotlib
samples = sorted(df['sample'].unique())
data = [df[df['sample'] == s]['corr'].values for s in samples]

plt.boxplot(data, positions=samples, patch_artist=True, boxprops=dict(facecolor="lightblue", alpha=0.5))

for i, s in enumerate(samples):
    y = df[df['sample'] == s]['corr'].values
    x = [s] * len(y)
    plt.scatter(x, y, color='red', alpha=0.7, zorder=3)

plt.title('Pre/Post Clean Correlation per Sample (p50 Full Exp)')
plt.xlabel('Sample (Concentration)')
plt.ylabel('Correlation (Cubic + Dilated Mask)')
plt.ylim(0, 1.0)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(r'C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\batch_correlation_plot.png')
print("Plot generated successfully")
