import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -6.081959061204903
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.75, learning_rate=0.1, loss="ls", max_depth=4, max_features=0.8, min_samples_leaf=3, min_samples_split=3, n_estimators=100, subsample=0.45)),
    StackingEstimator(estimator=SGDRegressor(alpha=0.0, eta0=1.0, fit_intercept=True, l1_ratio=0.5, learning_rate="invscaling", loss="epsilon_insensitive", penalty="elasticnet", power_t=0.0)),
    StackingEstimator(estimator=RandomForestRegressor(bootstrap=True, max_features=0.2, min_samples_leaf=17, min_samples_split=14, n_estimators=100)),
    MinMaxScaler(),
    LinearSVR(C=20.0, dual=False, epsilon=0.001, loss="squared_epsilon_insensitive", tol=1e-05)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
