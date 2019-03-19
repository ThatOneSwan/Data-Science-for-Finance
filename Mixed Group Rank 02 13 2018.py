#Implementation of the paper "Mixed group ranks: preference and confidence in classifier combination" to combine stock
#features

import pandas as pd
import numpy as np
import itertools
import scipy.stats as ss
from sklearn.linear_model import LogisticRegression
from scipy import sparse
import matplotlib.pyplot as plt
#load financial data into data frame, may have to change file path.  Also make sure the excel file is saved as a
# CSV file.  The original data is from a bloomberg terminal
stockdata = pd.read_csv("C:\Users\Jason\Desktop\Computational finance\CompFinResearch\Data\Hsu Stock Data(CSV).csv",
                        header=None)

#assigns specific feature info to specific dataframe
t = ['Date', 'IS_EPS', 'EBITDA', 'PROF_MARGIN', 'PE_RATIO', 'CASH_FLOW_PER_SH', 'PX_TO_BOOK_RATIO', 'CURR_ENTP_VAL',
     'PX_VOLUME', 'RETURN_COM_EQY']
x = stockdata.loc[~stockdata[0].isin(t)]
x = x.dropna(how='all')
y = stockdata.loc[stockdata[0] == 'Date']
y = y.iloc[0, :].tolist()[1:len(y)-1]
a = stockdata.loc[stockdata[0] == 'IS_EPS']
b = stockdata.loc[stockdata[0] == 'EBITDA']
c = stockdata.loc[stockdata[0] == 'PROF_MARGIN']
d = stockdata.loc[stockdata[0] == 'PE_RATIO']
e = stockdata.loc[stockdata[0] == 'CASH_FLOW_PER_SH']
f = stockdata.loc[stockdata[0] == 'PX_TO_BOOK_RATIO']
g = stockdata.loc[stockdata[0] == 'CURR_ENTP_VAL']
h = stockdata.loc[stockdata[0] == 'PX_VOLUME']
i = stockdata.loc[stockdata[0] == 'RETURN_COM_EQY']

attributes = [a, b, c, d, e, f, g, h, i]
avg_att = []

#formats and cleans data, removing
for n in attributes:
    j = n.drop(n.columns[0], axis=1)
    l = j.apply(pd.to_numeric, errors='coerce')
    h = l.mean(axis=1)
    avg_att.append(h)

data_arrays = []
for k in avg_att:
    j = map(float, k.as_matrix())
    l = np.asarray(j).T
    data_arrays.append(l)

names = x[0].as_matrix().T
data_matrix = np.column_stack(data_arrays)

#formatting data, removes nan values,
clean_data_matrix = data_matrix[~np.any(np.isnan(data_matrix), axis=1)]

#
clean_data_matrix = clean_data_matrix * np.array([1, 1, 1, -1, -1, 1, 1, -1, 1])
clean_names = names[~np.any(np.isnan(data_matrix), axis=1)]

data_vectors = []
for i in range(0, 9):
    j = clean_data_matrix[:, i]
    data_vectors.append(j)


#creating score functions from features
scores = []
for i in data_vectors:
    z = (1/(max(i)-min(i)))*(i-(min(i) * (np.ones(np.shape(i)))))
    scores.append(z)


#creating rank functions from features
ranks = []
for i in scores[0:9]:
    j = ss.rankdata(i, method="min")
    j = (j.shape[0])*(np.ones(np.shape(j))) - j + 1
    ranks.append(j)


top_5_scores = [scores[0], scores[2], scores[3], scores[5], scores[6]]
#Features: ACDFG
top_5_ranks = [ranks[0], ranks[2], ranks[3], ranks[5], ranks[6]]
top_5_ranks_int = []
for i in top_5_ranks:
    a = i.astype(int)
    top_5_ranks_int.append(a)


#combining using mixed group rank
rank_matrix = np.column_stack(top_5_ranks_int)


#obtain weights through logistic regression
model = LogisticRegression()
model.fit(rank_matrix, ranks[8])

weights = model.coef_

weights = np.split(weights, 5, axis=1)

rank_comb = []
weight_comb = []

for i in range(1, 6):
    a = list(itertools.combinations(top_5_ranks, i))
    b = list(itertools.combinations(weights, i))
    rank_comb = rank_comb + a
    weight_comb = weight_comb + b

mgr = []

for i in range(0, len(rank_comb)):
    j = []
    k = []

    for count in range(0, len(rank_comb[i])+1):
        g = list(itertools.combinations(rank_comb[i], count))
        h = list(itertools.combinations(weight_comb[i], count))
        j = j + g
        k = k + h

    j = [x for x in j if x != ()]
    k = [x for x in k if x != ()]
    k2 = []
    j2 = []

    for m in range(0, len(k)):
        v = reduce(lambda x, y: np.multiply(x, y), k[m])
        k2.append(v)

    for n in range(0, len(j)):
        x = np.column_stack(j[n])
        j2.append(x)

    rowlist = []

    for o in range(0, np.shape(j2[0])[0]):
        l = []
        for p in range(0, len(j2)):
            d = -1 * k2[p][o] * min(j2[p][o])
            l.append(d)
        t = sum(l)
        rowlist.append(t)
    rowMatrix = np.stack(rowlist)

    mgr.append(rowMatrix)

rank_mat = []


newScore = np.reshape(scores[8], (1903, 1))

for i in range(0, len(mgr)):
    j = np.concatenate([mgr[i], newScore], axis=1)
    rank_mat.append(j)

rank_sorts = []
for i in rank_mat:
    j = i[i[:, 0].argsort()]
    rank_sorts.append(j)

rank_perf = []

#since MGR is a score function
for i in rank_sorts:
    j = np.mean(i[:, 1][0:100])
    rank_perf.append(j)

comb_labels = []
for i in range(1, 6):
    a = list(itertools.combinations('ACDFG', i))
    comb_labels = comb_labels + a


labels = []
for i in comb_labels:
    a = ''.join(i)
    labels.append(a)

l = zip(labels, rank_perf)

one_feat = sorted(l[0:5], key=lambda x: x[1])
two_feat = sorted(l[5:15], key=lambda x: x[1])
three_feat = sorted(l[15:25], key=lambda x: x[1])
four_feat = sorted(l[25:30], key=lambda x: x[1])
five_feat = l[30]
k = one_feat + two_feat + three_feat + four_feat
k.append(five_feat)

[combinations, rank_y] = zip(*k)
combinations = list(combinations)

#plot effectiveness of different feature combinations
plt.plot(range(1, 32), rank_y, '-rx', label='rank combination', markersize=10, linestyle="--")
plt.xticks(range(1, 32), combinations, rotation=90)
plt.axvline(x=5, linestyle='--')
plt.axvline(x=15, linestyle='--')
plt.axvline(x=25, linestyle='--')
plt.axvline(x=30, linestyle='--')
plt.axhline(y = rank_y[combinations.index('G')], linestyle='--')
plt.xlabel('combinations')
plt.ylabel('performance')
#plt.title("Mixed Group Rank Combinations")
plt.legend(loc='lower right')
plt.tick_params('x', direction='out', length=15, top='off', pad=7)

plt.show()

















