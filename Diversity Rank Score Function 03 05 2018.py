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
clean_data_matrix = clean_data_matrix * np.array([1, 1, 1, -1, 1, 1, 1, -1, 1])
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

rankScoreFunctions = []
for i in range(0, 8):
    g = np.column_stack((ranks[i], scores[i]))
    h = g[g[:, 0].argsort()]
    rankScoreFunctions.append(h)

def div(x, y):
    z = x[:, 1] - y[:, 1]
    j = np.square(z)
    k = np.sum(j)
    l = np.sqrt(k)
    return l


att_pairs = list(itertools.combinations(rankScoreFunctions[0:8], 2))
label_pairs = list(itertools.combinations('ABCDEFGH', 2))
joined_labels = []

for i in label_pairs:
   a = ''.join(i)
   joined_labels.append(a)

diversity_score = []
for i in att_pairs:
    diversity = div(i[0], i[1])
    diversity_score.append(diversity)

diversity_rank = ss.rankdata(diversity_score, method="min")
diversity_rank = (diversity_rank.shape[0])*(np.ones(np.shape(diversity_rank))) - diversity_rank + 1

l = zip(joined_labels, diversity_score, diversity_rank)

l_sorted = sorted(l, key=lambda tup: tup[2])

[pairs_sort, div_score_sort, div_rank_sort] = zip(*l_sorted)

plt.plot(div_rank_sort, div_score_sort, '-bo')
plt.xlabel('Ranks')
plt.ylabel('Scores')
plt.axvline(x=20, linestyle='--')
plt.axvline(x=12, linestyle='--')
k = zip(pairs_sort, div_rank_sort, div_score_sort)
for label, x, y in k:
    plt.annotate(label, xy=(x, y), fontsize='12', va='bottom')
plt.show()


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



print "A: ", aScore
print "B: ", bScore
print "C: ", cScore
print "D: ", dScore
print "E: ", eScore
print "F: ", fScore
print "G: ", gScore
print "H: ", hScore





