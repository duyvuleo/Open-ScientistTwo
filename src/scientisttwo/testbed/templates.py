"""Audited reference implementations returned as text for isolated execution."""

import json


def reference_source(recipe):
    """The reference differs intentionally from incomplete/random upstream starters."""
    config = json.dumps({"task_id": recipe.task_id, "id": recipe.id_column,
                         "targets": recipe.targets, "metric": recipe.metric})
    return '''# ScientistTwo audited reference baseline; executes inside a container only.
import json
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CONFIG = json.loads(''' + repr(config) + ''')
seed = int(os.environ.get("SCIENTISTTWO_SEED", "0"))
train = pd.read_csv("/data/public/train.csv", dtype={CONFIG["id"]: str})
test = pd.read_csv("/data/public/evaluation.csv", dtype={CONFIG["id"]: str})
targets = CONFIG["targets"]
y = train[targets].to_numpy()
if CONFIG["task_id"] == "feedback":
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), max_features=30000), Ridge(alpha=1.0))
    model.fit(train["full_text"].fillna(""), y)
    predicted = model.predict(test["full_text"].fillna(""))
elif CONFIG["task_id"] == "amp-parkinsons-disease-progression-prediction":
    features = ["visit_month", "horizon", "updrs_number"]
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=seed, n_jobs=1)
    model.fit(train[features], y.ravel())
    predicted = np.maximum(model.predict(test[features]), 0).reshape(-1, 1)
else:
    features = [name for name in train.columns if name not in [CONFIG["id"], *targets]]
    numeric = [name for name in features if pd.api.types.is_numeric_dtype(train[name])]
    categorical = [name for name in features if name not in numeric]
    transforms = []
    if numeric:
        transforms.append(("numeric", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()), numeric))
    if categorical:
        transforms.append(("categorical", make_pipeline(SimpleImputer(strategy="most_frequent", keep_empty_features=True), OneHotEncoder(handle_unknown="ignore")), categorical))
    preprocessing = ColumnTransformer(transforms)
    estimator = LogisticRegression(max_iter=1000, random_state=seed) if CONFIG["metric"] == "accuracy" else Ridge(alpha=1.0)
    model = make_pipeline(preprocessing, estimator)
    model.fit(train[features], y.ravel())
    predicted = np.asarray(model.predict(test[features])).reshape(-1, 1)
result = pd.DataFrame(predicted, columns=targets)
result.insert(0, CONFIG["id"], test[CONFIG["id"]].to_numpy())
result.to_csv("submission.csv", index=False)
'''


def control_source(recipe):
    """A deterministic, deliberately simple comparison; its scores are never scripted."""
    source = reference_source(recipe)
    return source.replace("Ridge(alpha=1.0)", "Ridge(alpha=10.0)").replace(
        "LogisticRegression(max_iter=1000, random_state=seed)",
        "LogisticRegression(C=0.1, max_iter=1000, random_state=seed)",
    ).replace("min_samples_leaf=3", "min_samples_leaf=10")


def evaluator_source(recipe):
    config = json.dumps({"metric": recipe.metric, "id": recipe.id_column,
                         "targets": recipe.targets})
    return '''# Trusted scoring stage. This module never imports candidate code.
import json
from pathlib import Path
import pandas as pd
from metrics import score_frames

config = json.loads(''' + repr(config) + ''')
labels = pd.read_csv("/data/private/labels.csv", dtype={config["id"]: str})
prediction = pd.read_csv("/submission/submission.csv", dtype={config["id"]: str})
score = score_frames(config["metric"], labels, prediction, config["id"], config["targets"])
Path("/grade-output/grade.json").write_text(json.dumps({"valid_submission": True, "metrics": {config["metric"]: score}}, allow_nan=False))
'''
