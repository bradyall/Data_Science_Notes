import random
import warnings
from math import sqrt
import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors
from sklearn.model_selection import cross_validate, train_test_split
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import style
style.use('fivethirtyeight')


# KNN Classification hardcoded
#Key to KNN is Euclidean Distance. The algorithm compares the features of a y to all other y's features to find the k
# closest neighbors. Then it assigns y to the group with the majorty of the k's in the k closest neighbors

# Euclidean Distance = sqrt(sum from i to n(qi-pi)^2) where q is point 1 and p is point 2
# EX:  q=(1,3) p = (2,5)
    # sqrt((1-2)^2+(3-5)^2)

plot1 = [1,3]
plot2 = [2,5]

euclidean_distance = sqrt( (plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2)
print(euclidean_distance)

# Euclidean Distance in Numpy

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)

for i in dataset:
    for ii in dataset[i]:
        [plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]]
plt.scatter(new_features[0],new_features[1],color=result)
plt.show()

os.chdir('/Users/bropo/Desktop/Data Science/Machine Learning/youtube_series')
df = pd.read_csv('breast-cancer-wisconsin.data', names=['id',
                                                        'clump_thickness',
                                                        'unif_cell_size',
                                                        'unif_cell_shape',
                                                        'marg_adhesion',
                                                        'single_epith_cell_size',
                                                        'bare_nuclei',
                                                        'bland_chrom',
                                                        'norm_nucleoli',
                                                        'mitoses',
                                                        'class'])
df.replace('?',-99999, inplace=True)
df.drop(['id'],1,inplace=True)
full_data = df.astype(float).values.tolist()

random.shuffle(full_data)
test_size=0.2
train_set = {2:[],4:[]}
test_set = {2:[],4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1

print('Accuracy:', correct/total)
print(train_set)