#this script takes the financial data extracted from a bloomberg terminal.  The financial feature combinations are based on recommended combinations from "Average rank combination.py"

import pandas as pd
import numpy as np
import itertools
import scipy.stats as ss
#import matplotlib.pyplot as plt
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
#assigns each data frame for each feature to a specific variable
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

#Formats dataframe correctly
for n in attributes:
    j = n.drop(n.columns[0], axis=1)
    #k = j.dropna(axis=0, how='any')
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

#cleans data, removes nan values, and processes it
clean_data_matrix = data_matrix[~np.any(np.isnan(data_matrix), axis=1)]
clean_data_matrix = clean_data_matrix * np.array([1, 1, 1, -1, -1, 1, 1, -1, 1])
clean_names = names[~np.any(np.isnan(data_matrix), axis=1)]

data_vectors = []
for i in range(0, 9):
    j = clean_data_matrix[:, i]
    data_vectors.append(j)

#normalizes scores
scores = []
for i in data_vectors:
    a = (1/(max(i)-min(i)))*(i-(min(i) * (np.ones(np.shape(i)))))
    scores.append(a)

scores = []
for i in data_vectors:
    a = (1/(max(i)-min(i)))*(i-(min(i) * (np.ones(np.shape(i)))))
    scores.append(a)

#creates ranks for stocks based on score
ranks = []
for i in scores[0:8]:
    j = ss.rankdata(i, method="min")
    j = (j.shape[0])*(np.ones(np.shape(j))) - j + 1
    ranks.append(j)

#Recommended by Average Combinations
top_5_scores = [scores[0], scores[2], scores[3], scores[5], scores[6]]
#Features: ACDFG
#Recommended by Average Rank combinations
top_5_ranks = [ranks[0], ranks[2], ranks[3], ranks[5], ranks[6]]

score_comb = []
rank_comb = []

for i in range(1, 6):
    a = list(itertools.combinations(top_5_scores, i))
    score_comb = score_comb + a

for i in range(1, 6):
    a = list(itertools.combinations(top_5_ranks, i))
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
    j = np.mean(i[:, 1][1802:1903])
    score_perf.append(j)

for i in rank_sorts:
    j = np.mean(i[:, 1][0:100])
    rank_perf.append(j)

#l = zip(labels, score_perf, rank_perf)
comb_labels = []
for i in range(1, 6):
    a = list(itertools.combinations('ACDFG', i))
    comb_labels = comb_labels + a

labels = []
for i in comb_labels:
    a = ''.join(i)
    labels.append(a)

#Use combinations of features C and F, features C,D,F, and combinations of features A,C,D,F
CF = np.c_[clean_names, rank_mat[10]]
CDF = np.c_[clean_names, rank_mat[21]]
ACDF = np.c_[clean_names, rank_mat[25]]

pred1sort = CF[CF[:, 1].argsort()]
pred2sort = CDF[CDF[:, 1].argsort()]
pred3sort = ACDF[ACDF[:, 1].argsort()]

portfolio = []
for i in pred1sort[0:300]:
    if i in pred2sort[0:300]:
        if i in pred3sort[0:300]:
            portfolio.append(i[0])

CF_rank = CF[:,1]
CDF_rank = CDF[:, 1]
ACDF_rank = ACDF[:, 1]

agg_rank = np.mean((CF_rank, CDF_rank, ACDF_rank), axis=0)
agg_mat = np.c_[clean_names, agg_rank]

final_portfolio = []

for i in agg_mat:
    if i[0] in portfolio:
        final_portfolio.append(i)

[final_names, final_rank] = zip(*final_portfolio)

##normalized_scores = []
#for i in final_scores:
 #   j = (i - min(final_scores)) * (1/(max(final_scores)-min(final_scores)))
 #   normalized_scores.append(j)

portfolio_score = zip(final_names, final_rank)


portfolio_score.sort(key=lambda tup: tup[1])

best_portfolio = portfolio_score[0:99]

[stocknames, normal_scores] = zip(*best_portfolio)

final_weights = []
for i in normal_scores:
    j = i * (1/sum(normal_scores))
    final_weights.append(j)

#list stocks next to a score, higher score means higher quality

print investment = zip(stocknames, final_weights)
#print stocknames
#print list(reversed(final_weights))


