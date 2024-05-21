import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier
from tpot.export_utils import set_param_recursive
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('data/ToddlerAutismDataset.csv')
tpot_data = tpot_data.drop("Qchat-10-Score", axis=1)
features = tpot_data.drop('Class/ASD Traits', axis=1)
categorical_columns = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD', 'Who completed the test']
for i in categorical_columns:
    features[i] = label_encoder.fit_transform(features[i])
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, label_encoder.fit_transform(tpot_data['Class/ASD Traits']), random_state=42)

# Average CV score on the training set was: 0.9980582524271846
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LinearSVC(C=1.0, dual=False, loss="squared_hinge", penalty="l2", tol=1e-05)),
    XGBClassifier(learning_rate=0.5, max_depth=1, min_child_weight=17, n_estimators=100, n_jobs=1, subsample=1.0, verbosity=0)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

def tpot_pipeline():
    return exported_pipeline
