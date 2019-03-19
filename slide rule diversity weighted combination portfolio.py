from __future__ import division
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

clean_data_matrix = data_matrix[~np.any(np.isnan(data_matrix), axis=1)]
clean_data_matrix = clean_data_matrix * np.array([1, 1, 1, -1, -1, 1, 1, -1, 1])
clean_names = names[~np.any(np.isnan(data_matrix), axis=1)]

clean_data_matrix = np.c_[clean_names, clean_data_matrix]

data_vectors = []
for i in range(1, 10):
    j = clean_data_matrix[:, i]
    data_vectors.append(j)

scores = []
for i in data_vectors:
    a = (1/(max(i)-min(i)))*(i-(min(i) * (np.ones(np.shape(i)))))
    scores.append(a)

ranks = []
for i in scores[0:8]:
    j = ss.rankdata(i, method="min")
    j = (j.shape[0])*(np.ones(np.shape(j))) - j + 1
    ranks.append(j)

top_scores = [scores[0], scores[2], scores[3], scores[4], scores[6], scores[7]]
top_ranks = [ranks[0], ranks[2], ranks[3], ranks[4], ranks[6], ranks[7]]

div_score_mat = []
for i in top_scores:
    a = np.column_stack((i, scores[8]))
    div_score_mat.append(a)

div_rank_mat = []
for i in top_ranks:
    a = np.column_stack((i, scores[8]))
    div_rank_mat.append(a)

div_score_sort = []
for i in div_score_mat:
    j = i[i[:, 0].argsort()]
    div_score_sort.append(j)

div_rank_sort = []
for i in div_rank_mat:
    j = i[i[:, 0].argsort()]
    div_rank_sort.append(j)

div_score_perf = []
for i in div_score_sort:
    j = np.mean(i[:, 1][1803:1903])
    div_score_perf.append(j)

div_rank_perf = []
for i in div_rank_sort:
    j = np.mean(i[:, 1][0:100])
    div_rank_perf.append(j)


rankScoreFunctions = []
for i in range(0, 8):
    g = np.column_stack((scores[i], ranks[i]))
    h = g[g[:, 1].argsort()]
    rankScoreFunctions.append(h)


def div(x, y):
    z = x[:, 0] - y[:, 0]
    j = np.square(z)
    k = np.sum(j)
    l = np.sqrt(k)
    return l
att_pairs = list(itertools.combinations(rankScoreFunctions[0:8], 2))
label_pairs = list(itertools.combinations('ABCDEFGH', 2))

diversity_score = []

for i in range(0, len(att_pairs)):
    diversity = div(att_pairs[i][0], att_pairs[i][1])
    diversity_score.append(diversity)

label_pairs = list(itertools.combinations('ABCDEFGH', 2))
joined_labels = []

for i in label_pairs:
   a = ''.join(i)
   joined_labels.append(a)

diversity_rank = ss.rankdata(diversity_score, method="min")
diversity_rank = (diversity_rank.shape[0])*(np.ones(np.shape(diversity_rank))) - diversity_rank + 1

l = zip(joined_labels, diversity_score, diversity_rank)

l_sorted = sorted(l, key=lambda tup: tup[2])

[pairs_sort, div_score_sort, div_rank_sort] = zip(*l_sorted)

aScore = 0
bScore = 0
cScore = 0
dScore = 0
eScore = 0
fScore = 0
gScore = 0
hScore = 0

for i in range(0, 19):
    if 'A' in pairs_sort[i]:
        aScore = aScore + 1
    if 'B' in pairs_sort[i]:
        bScore = bScore + 1
    if 'C' in pairs_sort[i]:
        cScore = cScore + 1
    if 'D' in pairs_sort[i]:
        dScore = dScore + 1
    if 'E' in pairs_sort[i]:
        eScore = eScore + 1
    if 'F' in pairs_sort[i]:
        fScore = fScore + 1
    if 'G' in pairs_sort[i]:
        gScore = gScore + 1
    if 'H' in pairs_sort[i]:
        hScore = hScore + 1

diversity_strengths = [aScore, bScore, cScore, dScore, eScore, fScore, gScore, hScore]
top_div_strengths = [diversity_strengths[0], diversity_strengths[2], diversity_strengths[3], diversity_strengths[4],
                       diversity_strengths[6], diversity_strengths[7]]


#def div_s(x, l):
 #   s = []
  #  count = len(l)
   # z = 0
    #for i in range(0, len(l)):
     #   if i != x:
      #      s.append(i)
    #for i in s:
     #   d = div(x, i)
      #  z += d
    #z = z / count
    #return z




#div_strengths = []
#for i in range(0, 5):
 #   k = div_s(i, top_5_scores)
  #  div_strengths.append(k)


##div_score_pieces = []
##for i in range(0, 5):
  #  a = top_5_scores[i] * div_strengths[i]
   # div_score_pieces.append(a)

#div_rank_pieces =[]
#for i in range(0, 5):

inverted_div_ranks = []

for i in top_div_strengths:
    a = 1/i
    inverted_div_ranks.append(a)


score_comb = []
score_norms = []
rank_comb = []
rank_norms = []

for i in range(2, len(top_scores)+1):
    a = list(itertools.combinations(top_scores, i))
    b = list(itertools.combinations(top_div_strengths, i))
    score_comb = score_comb + a
    score_norms = score_norms + b

for i in range(2, len(top_scores)+1):
    a = list(itertools.combinations(top_ranks, i))
    b = list(itertools.combinations(inverted_div_ranks, i))
    rank_comb = rank_comb + a
    rank_norms = rank_norms + b

div_weighted_scores = []
for i in range(0, len(score_comb)):
    j =[a*b for a, b in zip(score_comb[i], score_norms[i])]
    k = sum(j)
    l = sum(score_norms[i])
    m = k * (1/l)
    div_weighted_scores.append(m)

div_weighted_ranks = []

for i in range(0, len(rank_comb)):
    j =[a*b for a, b in zip(rank_comb[i], rank_norms[i])]
    k = sum(j)
    l = sum(rank_norms[i])
    m = k * (1/l)
    n = (1/(max(m)-min(m)))*(m-(min(m) * (np.ones(np.shape(m)))))
    div_weighted_ranks.append(n)



score_mat = []
for i in div_weighted_scores:
    a = np.column_stack((i, scores[8]))
    score_mat.append(a)

rank_mat = []
for i in div_weighted_ranks:
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

comb_labels = []
for i in range(1, len(top_scores)+1):
    a = list(itertools.combinations('ACDEGH', i))
    comb_labels = comb_labels + a

labels = []
for i in comb_labels:
    a = ''.join(i)
    labels.append(a)

final_score_perf = div_score_perf + score_perf
final_rank_perf = div_rank_perf + rank_perf

l = zip(labels, final_score_perf, final_rank_perf)
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

ACE = np.c_[clean_names, score_mat[17]]
ADGH = np.c_[clean_names, score_mat[39]]
ADEGH = np.c_[clean_names, score_mat[55]]


pred1sort = ACE[ACE[:, 1].argsort()]
pred2sort = ADGH[ADGH[:, 1].argsort()]
pred3sort = ADEGH[ADEGH[:, 1].argsort()]

portfolio = []
for i in pred1sort[1002:1902]:
    if i in pred2sort[1002:1902]:
        if i in pred3sort[1002:1902]:
            portfolio.append(i[0])

ACE_score = ACE[:,1]
ADGH_score = ADGH[:, 1]
ADEGH_score = ADEGH[:, 1]

agg_score = np.mean((ACE_score, ADGH_score, ADEGH_score), axis=0)
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

best_portfolio = portfolio_score[0:40]

[lastnames, normal_scores] = zip(*best_portfolio)

final_weights = []
for i in normal_scores:
    j = i * (1/sum(normal_scores))
    final_weights.append(j)

investment = zip(lastnames, final_weights)
print lastnames
print final_weights
print len(lastnames)
