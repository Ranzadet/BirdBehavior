from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
X, y = make_regression(n_features=4, random_state=0)



regr = make_pipeline(StandardScaler(),
                      LinearSVR(random_state=0, tol=1e-5))
regr.fit(X, y)

print(regr.predict([[0,0,0,0]]))