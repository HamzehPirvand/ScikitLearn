import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

n_samples_1=1000
n_samples_2=100
centers=[[0.0, 0.0], [2.0, 2.0]]
clusters_std=[1.5, 0.5]
X, y=make_blobs(n_samples=[n_samples_1,n_samples_2],
                centers=centers,
                cluster_std=clusters_std,
                random_state=0, shuffle=False)

clf=svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)

wclf=svm.SVC(kernel='linear', class_weight={1: 10})
wclf.fit(X, y)


plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')

ax=plt.gca()
xlim=ax.get_xlim()
ylim=ax.get_ylim()


xx=np.linspace(xlim[0], xlim[1], 30)
yy=np.linspace(ylim[0], ylim[1], 30)
YY, XX=np.meshgrid(yy, xx)
xy=np.vstack([XX.ravel(), YY.ravel()]).T

Z=clf.decision_function(xy).reshape(XX.shape)

a=ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

Z=wclf.decision_function(xy).reshape(XX.shape)


b=ax.contour(XX, YY, Z, colors='r', levels=[0], alpha=0.5, linestyles=['-'])

plt.legend([a.collections[0], b.collections[0]], ["non weighted", "weighted"],
           loc="upper right")
plt.show()