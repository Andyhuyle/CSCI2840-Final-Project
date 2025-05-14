import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import numpy as np

"""
Script to run GWAS on a dataset
"""
# loads PLINK compatible .raw and .map files
raw_data = pd.read_csv("data/gwas_data.raw", sep=" ")
map_data = pd.read_csv("data/gwas_data.map", sep="\t", header=None, names=["CHR", "SNP", "CM", "BP"])

# extracts genotypes
genotype_cols = [col for col in raw_data.columns if col.startswith("SNP_")]
x = raw_data[genotype_cols].values  # genotype matrix -> (n_samples x n_SNPs)
y = raw_data["PHENOTYPE"].values  # phenotype -> (0=control, 1=case)

# runs GWAS (logistic regression used for binary traits)
p_values = []
beta_values = []

for snp in range(x.shape[1]):
    x_snp = sm.add_constant(x[:, snp])  # adds intercept
    model = sm.Logit(y, x_snp)

    result = model.fit(disp=0)  # fits model

    p_values.append(result.pvalues[1])  # p-value for a SNP
    beta_values.append(result.params[1])  # effect-size (beta)

gwas_results = pd.DataFrame({
    "SNP": map_data["SNP"],
    "CHR": map_data["CHR"],
    "BP": map_data["BP"],
    "P": p_values,
    "BETA": beta_values
})

# saves results to .csv file
data_dir = "GWAS_results"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
gwas_results.to_csv("GWAS_results/gwas_results.csv", index=False)


"""
Generates manhattan plot for visualization of GWAS_results
"""
# sorts by chromosome and base pair
gwas_results['CHR'] = gwas_results['CHR'].astype(str)
gwas_results = gwas_results.sort_values(['CHR', 'BP'])

# applies -log10(p) and indexing for plotting values
gwas_results['-log10p'] = -np.log10(gwas_results['P'])
gwas_results['ind'] = range(len(gwas_results))

# plot
fig, ax = plt.subplots(figsize=(12, 6))
colors = ['skyblue', 'navy']
x_labels = []
x_labels_pos = []

# groups chromosomes by plot
for i, (chromosome, group) in enumerate(gwas_results.groupby('CHR')):
    group.plot(kind='scatter', x='ind', y='-log10p', color=colors[i % len(colors)], ax=ax, s=10)

    x_labels.append(chromosome)
    x_labels_pos.append((group['ind'].iloc[0] + group['ind'].iloc[-1]) // 2)

# significance lines
plt.axhline(-np.log10(5e-8), color='red', linestyle='--', label='Genome-wide')
plt.axhline(-np.log10(1e-5), color='orange', linestyle='--', label='Suggestive')

ax.set_xticks(x_labels_pos)
ax.set_xticklabels(x_labels, rotation=90)
ax.set_xlabel('Chromosome')
ax.set_ylabel('-log10(p)')
ax.set_title('GWAS Manhattan Plot')
plt.legend()
plt.tight_layout()
plt.savefig("GWAS_results/gwas_manhattan.png")


"""
Finds the top 10 SNPs based on significance
"""
df = pd.read_csv("GWAS_results/gwas_results.csv")
df = df.dropna(subset=['P', 'BETA'])

# sorts by significance 
top_snps = df.sort_values('P').head(10)

# creates table
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')  

formatted_top_snps = top_snps.copy()
top_snps['Rank'] = top_snps['P'].rank(ascending=True).astype(int)
column_order = ['Rank', 'CHR', 'SNP', 'P', 'BETA']
top_snps = top_snps[column_order]

table = plt.table(
    cellText=top_snps.values,  
    colLabels=top_snps.columns, 
    cellLoc='center', 
    loc='center',  
    colColours=['#f3f3f3'] * len(top_snps.columns) 
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  
plt.title('GWAS Top 10 Most Significant SNPs', pad=20)
plt.savefig("GWAS_results/gwas_top_SNPs.png")
