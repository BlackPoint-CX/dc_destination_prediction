from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer
from sklearn.svm import SVR

import numpy as np

rng = np.random.RandomState(1)

def my_custom_loss_func(X_train_scaled, Y_train_scaled):
    error, M = 0, 0
    for i in range(0, len(Y_train_scaled)):
        z = (Y_train_scaled[i] - M)
        if X_train_scaled[i] > M and Y_train_scaled[i] > M and (X_train_scaled[i] - Y_train_scaled[i]) > 0:
            error_i = (abs(Y_train_scaled[i] - X_train_scaled[i]))**(2*np.exp(z))
        if X_train_scaled[i] > M and Y_train_scaled[i] > M and (X_train_scaled[i] - Y_train_scaled[i]) < 0:
            error_i = -(abs((Y_train_scaled[i] - X_train_scaled[i]))**(2*np.exp(z)))
        if X_train_scaled[i] > M and Y_train_scaled[i] < M:
            error_i = -(abs(Y_train_scaled[i] - X_train_scaled[i]))**(2*np.exp(-z))
    error += error_i
    return error

# Generate sample data
X = 5 * rng.rand(10000, 1)
y = np.sin(X).ravel()

# Add noise to targets
y[::5] += 3 * (0.5 - X.shape[0]/5)

train_size = 100

my_scorer = make_scorer(my_custom_loss_func, greater_is_better=True)

svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1),
                   scoring=my_scorer,
                   cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})

svr.fit(X[:train_size], y[:train_size])

print(svr.best_params_)
print(svr.score(X[train_size:], y[train_size:]))
