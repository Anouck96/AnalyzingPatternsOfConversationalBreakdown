import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pandas as pd

from matplotlib import rcParams

# figure size in inches
rcParams['figure.figsize'] = 12,6

def main():
    feat_df = pd.read_csv('features/all_data_anonymous_test.csv')
    # Remove the columns that do not have counts
    feat_df = feat_df.drop(columns=['grammar'])
    feat_df = feat_df.drop(columns=['full_s'])
    feat_df = feat_df.drop(columns=['beledig'])
    feat_df = feat_df.drop(columns=['eigennaam'])

    # Remove features that are not used due to lack of information
    feat_df = feat_df.drop(columns=['whites'])
    feat_df = feat_df.drop(columns=['test'])

    # Correct the 'pol' and 'subj columns by dividing values greater than 1 or less than -1 by 1000. (For items that are incorrectly saved/read)
    feat_df['pol'] = feat_df['pol'].apply(lambda x: x / 1000 if x > 1 or x < -1 else x)
    feat_df['subj'] = feat_df['subj'].apply(lambda x: x / 1000 if x > 1 or x < -1 else x)

    corr_matrix, p_values = spearmanr(feat_df)

    # Hide upper triangle of map
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # Create annotations with stars for significant p-values
    annot = np.empty_like(corr_matrix, dtype=object)
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            if p_values[i, j] <= 0.05:
                annot[i, j] = f"{corr_matrix[i, j]:.2f}*"
            else:
                annot[i, j] = f"{corr_matrix[i, j]:.2f}"



    columns = ['Word count', 'Character count', 'Sentence count', 'Polarity', 'Subjectivity', 'Reference words', 'Other', 'Repetitions', 'Loanwords', 'Readability', 'Typos', 'Outdated', 'Dyslexia', 'Form errors', 'Confused words', 'Unexpected', 'Commonness']
    figure = sns.heatmap(corr_matrix, mask=mask, annot=annot, fmt='', annot_kws={"size": 8}, xticklabels=columns, yticklabels=columns, cmap=sns.cubehelix_palette(reverse=True), cbar=False)

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig('correlation.pdf')
    plt.show()

    # Print the p-values
    print("P-values of the correlations:")
    print(pd.DataFrame(p_values, index=columns, columns=columns))

if __name__ == '__main__':
    main()
