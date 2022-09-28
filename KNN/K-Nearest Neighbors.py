#KNN through sklearn with Cancer Data

from math import sqrt
import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors
from sklearn.model_selection import cross_validate, train_test_split
import os
import numpy as np

accuracies = []
for i in range(25):
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

    X = np.array(df.drop(['class'],1))
    y = np.array(df['class'])

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=69)

    clf = neighbors.KNeighborsClassifier(n_jobs=-1)
    clf.fit(X_train,y_train)

    accuracy = clf.score(X_test,y_test)
    # #print(accuracy)
    #
    # example_measure = np.array([[4,2,1,1,1,2,3,2,1]])
    # example_measure = example_measure.reshape(len(example_measure),-1)
    #
    # prediction = clf.predict(example_measure)
    # #print(prediction)
    accuracies.append(accuracy)

print(sum(accuracies)/len(accuracies))


