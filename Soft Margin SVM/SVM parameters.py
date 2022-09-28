import numpy
import numpy as np
from sklearn import preprocessing, neighbors, svm
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

data_path = '/Users/bropo/Desktop/Data Science/Machine Learning/youtube_series/'
df = pd.read_csv(data_path+'breast-cancer-wisconsin.data', names=['id',
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
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

# Get support vectors themselves
support_vectors = clf.support_vectors_

# Visualize support vectors
plt.scatter(X_train[:,0], X_train[:,1])
plt.scatter(support_vectors[:,0], support_vectors[:,1], color='red')
plt.title('Linearly separable data with support vectors')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

print(support_vectors)