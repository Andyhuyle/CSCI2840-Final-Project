import numpy as np
import pandas as pd
from scipy.stats import norm
import os
from itertools import combinations

"""
Script to create simulated genotype data for GWAS
- features epistasis and dominance effects between SNPs (nonlinear effects)
"""
data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
                
n_samples = 1500        
n_snps = 50000          
minor_allele_freq = 0.2 
prevalence = 0.35       
noise_sd = 1.0 # environmental noise standard deviation

# generate family and individual IDs
family_ids = [f"FAM{i:03d}" for i in range(1, n_samples + 1)]
individual_ids = [f"IND{i:03d}" for i in range(1, n_samples + 1)]

# genotype matrix (0, 1, 2 for allele counts) 
genotypes = np.random.binomial(n=2, p=minor_allele_freq, size=(n_samples, n_snps))

# inject causal SNPs across multiple chromosomes
causal_chr_arr = [10, 9, 18]  
causal_snps_arr = [range(500, 505), range(8030, 8036), range(4782, 4797)] # indices of causal SNPs
flat_causal_snps = [snp for group in causal_snps_arr for snp in group]

# assigning effect sizes
effect_sizes = np.random.uniform(0.1, 0.2, size=len(flat_causal_snps))

# genetic risk scoring w/ additive model
risk = np.zeros(n_samples)
for i, snp in enumerate(flat_causal_snps):
    risk += genotypes[:, snp] * effect_sizes[i]

# genetic risk scoring w/ dominance model (nonlinear effect)
for i, snp in enumerate(flat_causal_snps):
    dominance = (genotypes[:, snp] == 1).astype(float)  # heterozygotes only
    risk += 0.5 * effect_sizes[i] * dominance

# pairwise epistasis (nonlinear effect)
allowed_epistatic_pairs = {(9,10)}  
causal_config = {
    10: range(500, 505), 
    9: range(8030, 8036),
    18: range(4782, 4797) 
}
chr_map = {snp: chr for chr, snp_range in causal_config.items() for snp in snp_range}

epistatic_pairs = []
for i, j in combinations(range(len(flat_causal_snps)), 2):
    snp1 = flat_causal_snps[i]
    snp2 = flat_causal_snps[j]
    chr1, chr2 = chr_map[snp1], chr_map[snp2]
    
    # checks if chr pair is allowed to interact
    if (chr1, chr2) in allowed_epistatic_pairs or (chr2, chr1) in allowed_epistatic_pairs:
        interaction_strength = np.random.uniform(0.15, 0.25)
        interaction = genotypes[:, snp1] * genotypes[:, snp2] * interaction_strength
        risk += interaction
        epistatic_pairs.append((snp1, snp2, chr1, chr2, interaction_strength))

# standardization of genetic risk w/ environmental noise
risk = (risk - np.mean(risk)) / np.std(risk)
noise = np.random.normal(0, noise_sd, size=n_samples)
liability = risk + noise  # realistic disease model that includes env noise

# assigns phenotypes
threshold = norm.ppf(1 - prevalence) 
phenotype = (liability > threshold).astype(int)

# case and controls
while np.mean(phenotype) < 0.1 or np.mean(phenotype) > 0.9:
    # If too extreme, add more noise and try again
    noise_sd *= 1.5
    noise = np.random.normal(0, noise_sd, size=n_samples)
    liability = risk + noise
    phenotype = (liability > threshold).astype(int)

sex = np.random.choice([1, 2], size=n_samples, p=[0.5, 0.5])
metadata = pd.DataFrame({
    'FID': family_ids,
    'IID': individual_ids,
    'PAT': 0,
    'MAT': 0,
    'SEX': sex,
    'PHENOTYPE': phenotype
})

snp_columns = [f"SNP_{i+1}" for i in range(n_snps)]
genotypes_df = pd.DataFrame(genotypes, columns=snp_columns)

plink_data = pd.concat([metadata, genotypes_df], axis=1)
plink_data.to_csv("data/gwas_data.raw", sep=" ", index=False, na_rep="NA")

# spaces out causal SNPs
chr_array = np.random.randint(1, 24, size=n_snps)
bp_array = np.cumsum(np.random.randint(100, 5000, size=n_snps)) + 1000000

for i, snp_range in enumerate(causal_snps_arr):
    chr_val = causal_chr_arr[i]
    for snp in snp_range:
        chr_array[snp] = chr_val

map_data = pd.DataFrame({
    'CHR': chr_array,
    'SNP': snp_columns,
    'CM': 0,
    'BP': bp_array
})
map_data.to_csv("data/gwas_data.map", sep="\t", header=False, index=False)

print(f"-> Causal SNPs: {len(flat_causal_snps)}")
print(f"-> Cases/Controls: {np.sum(phenotype)}/{len(phenotype)-np.sum(phenotype)}")
print("created PLINK .raw file: gwas_data.raw")
print("created PLINK .map file: gwas_data.map")

