import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans

X = np.array([[1,2],
              [2,3],
              [0.4,2],
              [8,7],
              [10,4],
              [6,6]])

clf = KMeans(n_clusters=2)
clf.fit(X)

centriods = clf.cluster_centers_
labels = clf.labels_

colors = ['g.','r.']

for i in range(len(X)):
    plt.plot(X[i][0],X[i][1], colors[labels[i]], markersize = 25)
plt.scatter(centriods[:,0], centriods[:,1], marker='x', s=150, linewidths=5)
plt.show()