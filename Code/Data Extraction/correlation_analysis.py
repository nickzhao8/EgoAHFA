import pandas as pd
import numpy as np
from pathlib import Path
import scipy

datafile = Path(r'M:\Wearable Hand Monitoring\CODE AND DOCUMENTATION\Nick Z\GRASSP_UEMS.csv')

data = pd.read_csv(datafile, sep=',')
grassp_mean = data['Mean score']
grassp_median = data['Median score']
uems = data['UEMS Score']

mean_pearson = scipy.stats.pearsonr(grassp_mean, uems, alternative='greater')
mean_spearman = scipy.stats.spearmanr(grassp_mean, uems, alternative='greater')
median_pearson = scipy.stats.pearsonr(grassp_median, uems, alternative='greater')
median_spearman = scipy.stats.spearmanr(grassp_median, uems, alternative='greater')

shapiro_mean = scipy.stats.shapiro(grassp_mean)
shapiro_median = scipy.stats.shapiro(grassp_median)
shapiro_uems = scipy.stats.shapiro(uems)

print(f'Spearman correlation coefficient(mean): {mean_spearman.correlation}; pvalue: {mean_spearman.pvalue}')
print(f'Pearson correlation coefficient(mean): {mean_pearson.statistic}; pvalue: {mean_pearson.pvalue}')
print(f'Spearman correlation coefficient(median): {median_spearman.correlation}; pvalue: {median_spearman.pvalue}')
print(f'Pearson correlation coefficient(median): {median_pearson.statistic}; pvalue: {median_pearson.pvalue}')
print(f'GRASSP mean score Shapiro-Wilk score: {shapiro_mean.statistic}; pvalue: {shapiro_mean.pvalue}')
print(f'GRASSP mean score Shapiro-Wilk score: {shapiro_median.statistic}; pvalue: {shapiro_median.pvalue}')
print(f'UEMS Shapiro-Wilk score: {shapiro_uems.statistic}; pvalue: {shapiro_uems.pvalue}')