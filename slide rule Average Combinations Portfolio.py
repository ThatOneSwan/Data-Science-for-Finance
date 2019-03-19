import pandas as pd
import numpy as np
import itertools
import scipy.stats as ss
import matplotlib.pyplot as plt
#load data into data frame, may have to change file path.  Also make sure the excel file is saved as a CSV file
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

clean_data_matrix = data_matrix[~np.any(np.isnan(data_matrix), axis=1)]
clean_data_matrix = clean_data_matrix * np.array([1, 1, 1, -1, -1, 1, 1, -1, 1])
clean_names = names[~np.any(np.isnan(data_matrix), axis=1)]

data_vectors = []
for i in range(0, 9):
    j = clean_data_matrix[:, i]
    data_vectors.append(j)


scores = []
for i in data_vectors:
    z = (1/(max(i)-min(i)))*(i-(min(i) * (np.ones(np.shape(i)))))
    scores.append(z)

ranks = []
for i in scores[0:8]:
    j = ss.rankdata(i, method="min")
    j = (j.shape[0])*(np.ones(np.shape(j))) - j + 1
    ranks.append(j)


top_scores = [scores[0], scores[2], scores[3], scores[4], scores[6], scores[7]]

#Features: ACDFG
top_ranks = [ranks[0], ranks[2], ranks[3], ranks[4], ranks[6], ranks[7]]

score_comb = []
rank_comb = []

for i in range(1, len(top_scores)+1):
    a = list(itertools.combinations(top_scores, i))
    score_comb = score_comb + a

for i in range(1, len(top_ranks)+1):
    a = list(itertools.combinations(top_ranks, i))
    rank_comb = rank_comb + a

score_avgs = []
for i in score_comb:
    j = np.mean(i, axis=0)
    score_avgs.append(j)

rank_avgs = []
for i in rank_comb:
    j = np.mean(i, axis=0)
    rank_avgs.append(j)

score_mat = []
rank_mat = []
for i in score_avgs:
    a = np.column_stack((i, scores[8]))
    score_mat.append(a)

for i in rank_avgs:
    a = np.column_stack((i, scores[8]))
    rank_mat.append(a)

score_sorts = []
for i in score_mat:
    j = i[i[:, 0].argsort()]
    score_sorts.append(j)


rank_sorts = []
for i in rank_mat:
    j = i[i[:, 0].argsort()]
    rank_sorts.append(j)

score_perf = []
rank_perf = []

for i in score_sorts:
    j = np.mean(i[:, 1][1803:1903])
    score_perf.append(j)

for i in rank_sorts:
    j = np.mean(i[:, 1][0:100])
    rank_perf.append(j)

#l = zip(labels, score_perf, rank_perf)
comb_labels = []
for i in range(1, len(top_scores)+1):
    a = list(itertools.combinations('ACDEGH', i))
    comb_labels = comb_labels + a

labels = []
for i in comb_labels:
    a = ''.join(i)
    labels.append(a)

l = zip(labels, score_perf, rank_perf)
one_feat = sorted(l[0:6], key=lambda x: x[1])
two_feat = sorted(l[6:21], key=lambda x: x[1])
three_feat = sorted(l[21:41], key=lambda x: x[1])
four_feat = sorted(l[41:56], key=lambda x: x[1])
five_feat = sorted(l[56:62], key=lambda x: x[1])
six_feat = l[62]
k = one_feat + two_feat + three_feat + four_feat + five_feat
k.append(six_feat)

[combinations, score_y, rank_y] = zip(*k)
combinations = list(combinations)

ACE = np.c_[clean_names, score_mat[22]]
CDGH = np.c_[clean_names, score_mat[53]]
ACDGH = np.c_[clean_names, score_mat[58]]


pred1sort = ACE[ACE[:, 1].argsort()]
pred2sort = CDGH[CDGH[:, 1].argsort()]
pred3sort = ACDGH[ACDGH[:, 1].argsort()]

portfolio = []
for i in pred1sort[1602:1902]:
    if i in pred2sort[1602:1902]:
        if i in pred3sort[1602:1902]:
            portfolio.append(i[0])

ACE_score = ACE[:,1]
CDGH_score = CDGH[:, 1]
ACDGH_score = ACDGH[:, 1]

agg_score = np.mean((ACE_score, CDGH_score, ACDGH_score), axis=0)
agg_mat = np.c_[clean_names, agg_score]

final_portfolio = []

for i in agg_mat:
    if i[0] in portfolio:
        final_portfolio.append(i)

[final_names, final_scores] = zip(*final_portfolio)

normalized_scores = []
for i in final_scores:
    j = (i - min(final_scores)) * (1/(max(final_scores)-min(final_scores)))
    normalized_scores.append(j)

portfolio_score = zip(final_names, normalized_scores)

portfolio_score.sort(key=lambda tup: tup[1])

best_portfolio = portfolio_score[46:86]

[lastnames, normal_scores] = zip(*best_portfolio)

final_weights = []
for i in normal_scores:
    j = i * (1/sum(normal_scores))
    final_weights.append(j)

investment = zip(lastnames, final_weights)
print lastnames
print final_weights
print len(lastnames)
