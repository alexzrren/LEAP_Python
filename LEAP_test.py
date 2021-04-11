# encoding: utf-8
"""
@author: ZiRui Ren
@contact: 1131061444@qq.com/renzirui@webmail.hzau.edu.cn
@time: 2021/3/22 13:25
@file: LEAP.py
@desc: 
"""
import pandas as pd
import numpy as np
from math import floor
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt

# read pseudotime csv
pseudo_Time = pd.read_csv('PseudoTime.csv', header=0, names=['cell', 'PseudoTime1', 'PseudoTime2'])
# fill in NA with 0
pseudo_Time = pseudo_Time.fillna(0)
# add pt1 with pt2 and get the sum as the pt
pseudo_Time['PseudoTime'] = pseudo_Time['PseudoTime1'] + pseudo_Time['PseudoTime2']
# sort the df of pseudotime to get the sorted cell name
pseudo_Time.sort_values(by=['PseudoTime', 'cell'], ascending=True, inplace=True)
# extract pesudotime data
cells = list(pseudo_Time['cell'])
# read expression matrix csv
raw_Matx = pd.read_csv('ExpressionData.csv', index_col=0, header=0)
# sort cell by pseudotime in the data
expr_Matx = raw_Matx[cells]

max_lag_prop = 1 / 3
symmetric = False

num_time_int = expr_Matx.shape[1]
max_lag = floor(num_time_int * (max_lag_prop))
window = num_time_int - max_lag
num_genes = expr_Matx.shape[0]

# Calculate Mean Vector and Covariance Matrix for Lag-0
data_0 = expr_Matx.iloc[:, 0:window]
means_0 = data_0.mean(1)
cent_0 = data_0.T - means_0
cor_0 = cent_0.corr(method='pearson')

# WARNING: small float sum may cause accumulative error
rowsumx_0 = cent_0.apply(lambda x: x.sum())
rowsumx2_0 = cent_0.apply(lambda x: (x ** 2).sum())

# Calculate Cross-Correlation for Lags
max_lag_cor = cor_0
lag_max = np.zeros([num_genes, num_genes], dtype=int) - 1

lag_max_record = np.zeros([num_genes, num_genes], dtype=int)

for i in tqdm(range(0, max_lag)):
    # slide lag window and calculate MAC values
    data_lag = expr_Matx.iloc[:, i + 1:window + i + 1]
    means_lag = data_lag.mean(1)
    cent_lag = data_lag.T - means_lag
    cent_lagint = cent_lag

    rowsumx_lag = cent_lag.sum()
    rowsumx2_lag = cent_lag.apply(lambda x: (x ** 2).sum())

    cor_lag_num = np.dot(cent_0.T, cent_lag) - np.dot((1 / window) * rowsumx_0.T, rowsumx_lag)
    cor_lag_deno1 = pd.DataFrame(np.sqrt(rowsumx2_0 - rowsumx_0 ** 2 / window))
    cor_lag_deno2 = pd.DataFrame(np.sqrt(rowsumx2_lag - rowsumx_lag ** 2 / window))
    cor_lag_deno = np.dot(cor_lag_deno1, cor_lag_deno2.T)
    cor_lag = cor_lag_num / cor_lag_deno

    # Keep cor only if greater than previous lag cor
    temp = max_lag_cor
    max_lag_cor = np.where(np.abs(max_lag_cor) < np.abs(cor_lag), cor_lag, max_lag_cor)
    lag_max = np.where(np.abs(temp) < np.abs(cor_lag), lag_max_record, lag_max)
    lag_max_record += 1

genes = cent_0.columns.values
MACs_greatest = max_lag_cor
lag_greatest = lag_max

# if symmetric, get the value which has greater absolute value
if symmetric:
    max_lag_cor_t = max_lag_cor.T
    lag_max_t = lag_max.T
    MACs_greatest = np.maximum(max_lag_cor, max_lag_cor_t)
    # the same is for the lag data
    lag_greatest = np.where(np.abs(max_lag_cor) < np.abs(max_lag_cor_t), lag_max_t, lag_max)
    
# transform the matrix into triplet for latter steps
shape = MACs_greatest.shape[0]
genes = cent_0.columns.values
Triplet = []
for row in range(shape):
    for col in range(shape):
        if row != col:
            Triplet.append([genes[row], genes[col], MACs_greatest[row, col]])
TripletMatx = pd.DataFrame(Triplet, columns=['gene1', 'gene2', 'value'])

refNetwork = pd.read_csv('refNetwork.csv', header=0)
sortedrefNetwork = refNetwork.sort_values(by=['Gene1', 'Gene2'], ascending=[True, True])
ref = sortedrefNetwork.itertuples()
curr_ref = next(ref)
print('\n')
values = []
labels = []

# generate list of values and its label for drawing roc curve
for row in TripletMatx.itertuples():
    if curr_ref[1] == curr_ref[2]:
        try:
            curr_ref = next(ref)
            continue
        except StopIteration:
            print('STOP')
            break
    if curr_ref[1] == row[1] and curr_ref[2] == row[2]:
        values.append(row[3])
        labels.append(1)
        try:
            curr_ref = next(ref)
            continue
        except StopIteration:
            print('DONE')
            break
    else:
        values.append(row[3])
        labels.append(0)


# drawing roc curve and calculate auc value
fpr, tpr, thresholds = metrics.roc_curve(labels, values, pos_label=1, sample_weight=None)
auc_value = metrics.auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (AUC = %0.2f)' % auc_value)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
print('AUC_value=', auc_value)
