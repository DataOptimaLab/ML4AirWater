import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -1.8696589710740739
exported_pipeline = make_pipeline(
    MinMaxScaler(),
    Normalizer(norm="l1"),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.95, learning_rate=0.1, loss="lad", max_depth=9, max_features=0.9000000000000001, min_samples_leaf=5, min_samples_split=6, n_estimators=100, subsample=0.9500000000000001)),
    MaxAbsScaler(),
    KNeighborsRegressor(n_neighbors=2, p=1, weights="distance")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
