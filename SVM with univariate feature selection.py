import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


X, y=load_iris(return_X_y=True)
np.random.seed(0)
X=np.hstack((X, 2 * np.random.random((X.shape[0], 36))))

a=Pipeline([('anova', SelectPercentile(chi2)),
            ('scaler',StandardScaler()),
            ('svc', SVC(gamma="auto"))])

score_means=list()
score_stds=list()
percentiles=(1,3,6,10,15,20,30,40,60,80,100)

for percentile in percentiles:
    a.set_params(anova__percentile=percentile)
    this_scores=cross_val_score(a, X, y)
    score_means.append(this_scores.mean())
    score_stds.append(this_scores.std())

plt.errorbar(percentiles, score_means, np.array(score_stds))
plt.title('Performance of the SVM-Anova varying the percentile of features selected')
plt.xticks(np.linspace(0, 100, 11, endpoint=True))
plt.xlabel('Percentile')
plt.ylabel('Accuracy Score')
plt.axis('tight')
plt.show()