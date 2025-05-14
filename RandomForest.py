import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from kneed import KneeLocator
import time
import logging
import os

# logger bc data was taking too long to load and model too long to traing lol :(
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

data_dir = "rf_results"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

class PythonRandomForest:
    def __init__(self, raw_file, map_file):
        """
        sklearn library Random Forest
        
        args:
            raw_file: path to PLINK .raw file
            map_file: path to PLINK .map file
        """
        self.raw_file = raw_file
        self.map_file = map_file
        self.test_size = 0.2 # 20% of data is used for testing, 80% for training
        self.random_state = 100 # random seed
        self.x = None
        self.y = None
        self.snp_names = None
        self.map_df = None
        self.rf = None
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
    

    def ld_prune(self, x, r2_threshold=0.9):
        """prunes SNPs with R² > threshold"""
        R_squared = np.corrcoef(x.T)  # pearson correlation (R^2)
        np.fill_diagonal(R_squared, 0)  
        
        to_remove = set()
        for i in range(R_squared.shape[0]):
            logger.info(f"{i}")
            if i not in to_remove:
                high_ld = np.where(R_squared[i] > r2_threshold)[0]
                to_remove.update(high_ld[high_ld != i])  # removes high LD SNP
        
        return np.setdiff1d(np.arange(x.shape[1]), list(to_remove))


    def load_data(self):
        """load and preprocess PLINK .raw and .map files"""
        logger.info("Loading data...")
        
        # loads phenotype and genotype data
        plink_raw = pd.read_csv(self.raw_file, sep='\s+', dtype={'PHENOTYPE': float})
        
        # extracts phenotypes (serves as y)
        self.y = plink_raw['PHENOTYPE'].values
        
        # extract genotypes 
        snp_cols = [col for col in plink_raw.columns if col.startswith('SNP_')]
        self.x = plink_raw[snp_cols].astype(float).values
        
        # LD pruning 
        logger.info("Pruning High LD SNPs...")
        keep_indices = self.ld_prune(x=self.x, r2_threshold=0.9)
        self.x = self.x[:, keep_indices]

        # loads SNP info into dataframe
        self.map_df = pd.read_csv(self.map_file, sep='\s+', header=None,
                                names=['CHR', 'SNP', 'CM', 'BP'])
        self.snp_names = self.map_df['SNP'].values[keep_indices]
        
        # split data for training and testing
        logger.info("Splitting Data for Training and Testing...")
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=self.test_size, random_state=self.random_state)

        logger.info(f"Data loaded w/ {self.x.shape[1]} SNPs and {self.x.shape[0]} samples")
            
    
    def train_model(self, ntree=1000, mtry=0.1, 
                   min_samples_leaf=5, class_weight='balanced'):
        """
        trains RF model
        
        args:
            ntree: num of trees (default is 1000 as mentioned in paper)
            mtry: features to consider at each split (default 0.1 as mentioned in paper)
            min_samples_leaf: min samples per leaf (default 5)
            class_weight: class weights (default 'balanced' as mentioned in the paper)
        """
        logger.info("Training python RF model w/ adaptive sparsity pruning...")
        start_time = time.time()
        
        best_oob = -np.inf
        best_indices = np.arange(self.x_train.shape[1])
        x_current = self.x_train.copy()
        oob_history = []
        n_snps_history = []
        
        # only punes SNPs in 4 iterations or less otherwise OOB skyrockets :(
        for i in range(4):
            # train RF with current SNPs
            self.rf = RandomForestClassifier(
                criterion="gini",
                n_estimators=ntree,
                max_features=mtry,
                min_samples_leaf=min_samples_leaf,
                class_weight=class_weight,
                oob_score=True,
                n_jobs=-1,
                random_state=self.random_state
            )
            self.rf.fit(x_current, self.y_train)
            
            current_oob = self.rf.oob_score_
            oob_history.append(current_oob)
            n_snps_history.append(x_current.shape[1])
            
            logger.info(f"Iteration {i+1}: {x_current.shape[1]} SNPs | OOB: {current_oob:.3f}")
            
            # checks for OOB score stabilization
            if i > 0:
                oob_change = abs(oob_history[-1] - oob_history[-2])
                if oob_change < 0.001:
                    logger.info(f"OOB stabilized (Δ < {0.001}): stopping pruning!")
                    break
            
            # updates best model if improved
            if current_oob > best_oob:
                best_oob = current_oob
                best_indices = np.arange(x_current.shape[1])
            
            # prune weakest SNPs (except on last iteration)
            if i < 3:
                importances = self.rf.feature_importances_
                sorted_idx = np.argsort(importances)[::-1]
                n_keep = int(len(sorted_idx) * (1 - 0.5))
                x_current = x_current[:, sorted_idx[:n_keep]]
        
        # trains final model on best SNP set
        self.rf.fit(self.x_train[:, best_indices], self.y_train)
        self.important_snp_indices = best_indices
        self.x_train_pruned = self.x_train[:, best_indices]
        self.x_test_pruned = self.x_test[:, best_indices]

        train_time = time.time() - start_time
        logger.info(f"Training completed in {train_time:.2f} seconds")
        logger.info(f"OOB Score: {self.rf.oob_score_:.3f}")
        logger.info(f"Test Accuracy: {self.rf.score(self.x_test_pruned, self.y_test):.3f}")
            
    
    def analyze_results(self, top_n=10):
        """
        analysze/visualize feature importances
        
        args:
            top_n = top number of SNPs to display
        """
        # gets gini (importance) scores 
        importances = self.rf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.rf.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]
        
        results_df = pd.DataFrame({
            'SNP': self.snp_names[indices],
            'Importance': importances[indices],
            'Std': std[indices]
        }).merge(self.map_df, on='SNP')
        results_df['Rank'] = np.arange(1, len(results_df)+1)
        
        # scree plot
        # auto-'elbow' detection
        knee = KneeLocator(
            results_df['Rank'], 
            results_df['Importance'],
            curve='convex',
            direction='decreasing',
            interp_method='polynomial'
        )
        elbow_rank = knee.elbow

        plt.figure(figsize=(12, 6))
        plt.plot(results_df['Rank'], results_df['Importance'], 'b-', label='SNP GINI Importance')
        plt.axvline(x=elbow_rank, color='r', linestyle='--', 
                    label=f'Elbow (Top {elbow_rank} SNPs)')
        plt.xlabel('SNP Rank')
        plt.ylabel('Gini Importance')
        plt.title('Random Forest SNP Importance Scree Plot\n(Elbow Method for Cutoff)')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('rf_results/python_rf_importance_scree.png', dpi=300)

        # top SNPs table
        top_10_snps = results_df.head(top_n).copy()
        fig, ax = plt.subplots(figsize=(top_n, 4))
        ax.axis('off')

        table_data = top_10_snps[['Rank', 'CHR', 'SNP', 'Importance']]
        table_data.columns = ['Rank', 'Chr', 'SNP', 'RF Importance']

        table = ax.table(
            cellText=table_data.values,
            colLabels=table_data.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.1, 0.3, 0.3, 0.3]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        plt.title('Top 10 SNPs by GINI Importance Score', pad=20)
        plt.tight_layout()
        plt.savefig('rf_results/top_snps_table.png', dpi=300, bbox_inches='tight')

        # manhattan plot (used for visualization only, not really comparable to GWAS)
        plot_results = results_df.copy()
        plot_results['CHR'] = plot_results['CHR'].astype(str)
        plot_results = plot_results.sort_values(['CHR', 'BP'])
        plot_results['ind'] = range(len(plot_results))
        
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        x_labels = []
        x_labels_pos = []
        colors = ['skyblue', 'navy']

        # groups chromosomes by plot
        for i, (chromosome, group) in enumerate(plot_results.groupby('CHR')):
            group.plot(kind='scatter', x='ind', y='Importance', color=colors[i % len(colors)], ax=ax2, s=10)

            x_labels.append(chromosome)
            x_labels_pos.append((group['ind'].iloc[0] + group['ind'].iloc[-1]) // 2)

        ax2.set_xticks(x_labels_pos)
        ax2.set_xticklabels(x_labels, rotation=90)
        ax2.set_xlabel("Chromosome")
        ax2.set_ylabel("Importance Score")
        ax2.set_title("RF Manhattan Plot")
        plt.tight_layout()
        plt.savefig("rf_results/rf_gwas_manhattan.png")

        return results_df
            

def main():
    analyzer = PythonRandomForest(
        raw_file="data/gwas_data.raw",
        map_file="data/gwas_data.map"
    )
    
    logger.info("Loading data...")
    analyzer.load_data()

    logger.info("Training model...")
    analyzer.train_model(
        ntree=1000,
        mtry=0.1, 
        min_samples_leaf=5
    )
    
    results = analyzer.analyze_results(top_n=10)
    results.to_csv("rf_results/python_rf_gwas_results.csv", index=False)
    

if __name__ == "__main__":
    main()