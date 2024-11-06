import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mode,stats, ttest_ind
import pandas as pd
from statsmodels.stats.multitest import multipletests
from numpy import mean, var, sqrt


def cohen_d(x, y):
    """Calculate Cohen's d for effect size. Based on https://machinelearningmastery.com/effect-size-measures-in-python/"""
    # Sizes of repair/norepair
    nr = len(x)
    nn = len(y)
    # Variance
    varr, varn = var(x, ddof=1), var(y, ddof=1)
    # Pooled standard deviation
    pooled_sd = sqrt(((nr-1) * varr + (nn - 1) * varn) / (nr + nn - 2))
    # Mean
    meanr, meann = mean(x), mean(y)
    # Return effect size
    return (meanr - meann) / pooled_sd


def main():
    # Read in both files
    feat_df_repairs = pd.read_csv('features/all_data_anonymous_test.csv')
    feat_df_norepairs = pd.read_csv('features/nonrepairs_features_nobuttons_anon.csv')

    # Correct the 'pol' and 'subj columns by dividing values greater than 1 or less than -1 by 1000. (For items that are incorrectly saved/read)
    feat_df_repairs['pol'] = feat_df_repairs['pol'].apply(lambda x: x / 1000 if x > 1 or x < -1 else x)
    feat_df_repairs['subj'] = feat_df_repairs['subj'].apply(lambda x: x / 1000 if x > 1 or x < -1 else x)

    # Correct the 'pol' and 'subj columns by dividing values greater than 1 or less than -1 by 1000. (For items that are incorrectly saved/read)
    feat_df_norepairs['pol'] = feat_df_norepairs['pol'].apply(lambda x: x / 1000 if x > 1 or x < -1 else x)
    feat_df_norepairs['subj'] = feat_df_norepairs['subj'].apply(lambda x: x / 1000 if x > 1 or x < -1 else x)

    # Drop columns that are not used for analysis due to lack of information
    feat_df_repairs = feat_df_repairs.drop(columns=['whites'])
    feat_df_norepairs = feat_df_norepairs.drop(columns=['whites'])
    feat_df_repairs = feat_df_repairs.drop(columns=['test'])
    feat_df_norepairs = feat_df_norepairs.drop(columns=['test'])

    # Calculate mean and standard deviation for each column of feat repairs and feat no repairs
    means = feat_df_repairs.mean()
    std_devs = feat_df_repairs.std()
    means_norepair = feat_df_norepairs.mean()
    std_devs_norepair = feat_df_norepairs.std()

    # Calculate frequency of non-zero items for each column
    freq_repairs = (feat_df_repairs != 0).sum()
    freq_norepairs = (feat_df_norepairs != 0).sum()


    mean_difference = means - means_norepair

    # Ratio of frequencies (non-zero items)
    freq_ratios = freq_repairs / freq_norepairs


    # Compare means of features (t-test)
    t_test_results = {column: ttest_ind(feat_df_repairs[column], feat_df_norepairs[column], nan_policy='omit').pvalue
                      for column in feat_df_repairs.columns}


    #Holm-Bonferroni correction with statsmodels
    p_values = list(t_test_results.values())
    reject, adjusted_p_values, _, _ = multipletests(p_values, method='holm')
    adjusted_t_test_results = dict(zip(t_test_results.keys(), adjusted_p_values))

    # Calculate effect size (Cohen's d)
    effect_sizes = {column: cohen_d(feat_df_repairs[column].dropna(), feat_df_norepairs[column].dropna())
                    for column in feat_df_repairs.columns}

    # Split columns into two halves at 'ref'
    columns = list(means.index)
    split_index = columns.index('ref') + 1
    columns1 = columns[:split_index]
    columns2 = columns[split_index:]

    # Make LaTeX table
    latex_table = "\\begin{table}[h!]\n\\centering\n\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
    latex_table += "Feature & Mean & SD & Mean Difference & Frequency Ratio & Effect Size \\\\\n\\midrule\n"

    for col in means.index:
        mean_diff_str = f"{round(mean_difference[col],3)}"
        if adjusted_t_test_results[col] <= 0.05:
            mean_diff_str += "*"
        freq_ratio_str = f"{round(freq_ratios[col],3)}"
        effect_size_str = f"{round(effect_sizes[col],3)}"
        latex_table += f"{col} & {round(means[col],3)} & {round(std_devs[col],3)} & {mean_diff_str} & {freq_ratio_str} & {effect_size_str} \\\\\n"

    latex_table += "\\bottomrule\n\\end{tabular}\n\\caption{Means, standard deviations, mean differences, frequency ratios, and effect sizes between repair and control data (* if p-value <=0.05).}\n\\end{table}"

    print(latex_table)
if __name__== '__main__':
    main()
