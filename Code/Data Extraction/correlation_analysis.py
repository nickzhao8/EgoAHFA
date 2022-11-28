import pandas as pd
import numpy as np
from pathlib import Path
import scipy

datafile = Path(r'M:\Wearable Hand Monitoring\CODE AND DOCUMENTATION\Nick Z\GRASSP_UEMS.csv')

data = pd.read_csv(datafile, sep=',')
grassp_mean = data['Mean score']
grassp_median = data['Median score']
uems = data['UEMS Score']

pearson = scipy.stats.pearsonr(grassp_mean, uems, alternative='greater')
spearman = scipy.stats.spearmanr(grassp_mean, uems, alternative='greater')

shapiro_mean = scipy.stats.shapiro(grassp_mean)
shapiro_uems = scipy.stats.shapiro(uems)

print(f'Spearman correlation coefficient: {spearman.correlation}; pvalue: {spearman.pvalue}')
print(f'Pearson correlation coefficient: {pearson.statistic}; pvalue: {pearson.pvalue}')
print(f'GRASSP mean score Shapiro-Wilk score: {shapiro_mean.statistic}; pvalue: {shapiro_mean.pvalue}')
print(f'UEMS Shapiro-Wilk score: {shapiro_uems.statistic}; pvalue: {shapiro_uems.pvalue}')