"""Declarative, benchmark-independent CSV development protocols.

Only trusted preparation and scoring execute here. Baseline source is returned as
text and executes through the isolated runner, exactly like model proposals.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from ..benchmarks.files import file_hash, tree_manifest
from ..benchmarks.models import relative_path
from .evaluation import metric_specs
from .models import Candidate, SourceFile


REGRESSION = {"rmse", "mse", "mae", "r2", "smape"}
CLASSIFICATION = {"roc_auc", "log_loss", "accuracy", "macro_f1", "f1", "dice"}


class CSVDevelopment:
    def __init__(self, native, config):
        self.native, self.config = native, config
        self.settings = config.settings
        if config.train_file == config.test_file:
            raise ValueError("Training and prediction input files must differ")
        self.frame = pd.read_csv(native.public_dir / config.train_file)
        if config.target not in self.frame or self.frame[config.target].isna().any():
            raise ValueError("Configured target missing or containing null training targets")
        self.labels = self.frame[config.target].to_numpy()
        self.problem_type = self.settings.get(
            "problem_type", "regression" if config.metric in REGRESSION else "classification"
        )
        if self.problem_type not in {"classification", "regression"}:
            raise ValueError("problem_type must be classification or regression")
        if self.problem_type == "regression":
            self.labels = pd.to_numeric(self.frame[config.target], errors="raise").to_numpy()
            if not np.isfinite(self.labels).all():
                raise ValueError("Regression targets must be finite numeric values")
        metrics = set(metric_specs(config))
        supported = REGRESSION if self.problem_type == "regression" else CLASSIFICATION
        if not metrics <= supported:
            raise ValueError("CSV adapter does not support these metrics for its problem_type")
        # Probability and label predictions require different submission contracts.
        if metrics & {"roc_auc", "log_loss"} and len(metrics) > 1:
            raise ValueError("Probability metrics require a single declared metric")
        if config.metric == "roc_auc" and set(self.labels) != {0, 1}:
            raise ValueError("ROC AUC adapter requires binary 0/1 targets")
        if config.metric == "log_loss" and set(map(str, self.labels)) != set(config.classes):
            raise ValueError("Configured classes differ from training labels")
        if config.metric == "log_loss" and len(config.classes) != len(config.prediction_columns):
            raise ValueError("Declare one prediction column per class in class order")
        if metrics & {"f1", "dice"} and set(self.labels) != {0, 1}:
            raise ValueError("Binary F1/Dice requires binary 0/1 targets; use macro_f1 otherwise")
        if config.metric != "log_loss" and len(config.prediction_columns) != 1:
            raise ValueError("Scalar/label prediction requires one prediction column")
        self.features = self.settings.get("features", {"text": [config.text_column]})
        if not isinstance(self.features, dict) or not set(self.features) <= {
            "text",
            "numeric",
            "categorical",
        }:
            raise ValueError("features maps text, numeric, and categorical to column lists")
        columns = []
        for kind, names in self.features.items():
            if not isinstance(names, list) or any(not isinstance(x, str) for x in names):
                raise ValueError("Feature groups must be lists of column names")
            columns.extend(names)
        if not columns or len(columns) != len(set(columns)):
            raise ValueError("Declare nonempty, distinct feature columns")
        if config.target in columns or not set(columns) <= set(self.frame):
            raise ValueError("Features must exist and cannot include the target")
        self.id_columns = self.settings.get("id_columns", [])
        if (
            not isinstance(self.id_columns, list)
            or any(not isinstance(x, str) for x in self.id_columns)
            or len(self.id_columns) != len(set(self.id_columns))
            or not set(self.id_columns) <= set(self.frame)
        ):
            raise ValueError("id_columns must be distinct existing training columns")
        if set(self.id_columns) & (set(columns) | {config.target}):
            raise ValueError("Submission identifiers cannot be features or targets")
        if self.id_columns and (
            self.frame[self.id_columns].isna().any().any()
            or self.frame.duplicated(self.id_columns).any()
        ):
            raise ValueError("Submission identifiers must be non-null and unique")
        for column in self.features.get("numeric", []):
            pd.to_numeric(self.frame[column], errors="raise")
        sample = self.settings.get("sample_submission")
        if sample is not None:
            sample = relative_path(sample)
            samples = [native.public_dir / sample]
        else:
            samples = sorted(native.public_dir.glob("sample*.csv"))
        declared = self.settings.get("submission_columns")
        if declared is None:
            if len(samples) != 1:
                raise ValueError("Declare submission_columns or one sample_submission CSV")
            declared = list(pd.read_csv(samples[0], nrows=0).columns)
        if (
            not isinstance(declared, list)
            or any(not isinstance(x, str) for x in declared)
            or len(declared) != len(set(declared))
            or not set(config.prediction_columns + self.id_columns) <= set(declared)
            or not set(declared) - set(config.prediction_columns) <= set(self.frame)
            or config.target in set(declared) - set(config.prediction_columns)
        ):
            raise ValueError("Submission columns must contain predictions and available metadata")
        self.submission_columns = declared
        self.sample_paths = [p.relative_to(native.public_dir).as_posix() for p in samples]
        self.prediction_file = relative_path(
            self.settings.get("prediction_file", getattr(native, "submission", "submission.csv"))
        )
        if self.prediction_file in {config.train_file, config.test_file, *self.sample_paths}:
            raise ValueError("Prediction path cannot replace an input file")
        self.split = self.settings.get(
            "split", {"kind": "random" if self.problem_type == "regression" else "stratified"}
        )
        self.train, self.valid = self._partition(
            np.arange(len(self.frame)), config.validation_fraction
        )
        if config.subset_fraction < 1:
            self.subset, _ = self._partition(self.train, 1 - config.subset_fraction)
        else:
            self.subset = self.train.copy()
        if self.problem_type == "classification" and (
            set(self.labels[self.train]) != set(self.labels)
            or set(self.labels[self.subset]) != set(self.labels)
        ):
            raise ValueError("Every training partition must contain all declared classes")
        if config.metric == "roc_auc" and len(set(self.labels[self.valid])) < 2:
            raise ValueError("ROC AUC validation requires both classes")

    def _partition(self, rows, fraction):
        if not isinstance(self.split, dict) or set(self.split) - {"kind", "column"}:
            raise ValueError("split accepts kind and optional column")
        kind = self.split.get("kind")
        c = self.config
        if kind in {"random", "stratified"}:
            if kind == "stratified" and self.problem_type != "classification":
                raise ValueError("Stratification requires classification targets")
            return train_test_split(
                rows,
                test_size=fraction,
                random_state=c.split_seed,
                stratify=self.labels[rows] if kind == "stratified" else None,
            )
        column = self.split.get("column")
        if column not in self.frame or column == c.target or self.frame[column].isna().any():
            raise ValueError("Group/time splits require a non-null non-target column")
        values = self.frame.iloc[rows][column]
        if kind == "group":
            train, valid = next(
                GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=c.split_seed).split(
                    rows, groups=values
                )
            )
            return rows[train], rows[valid]
        if kind == "time":
            times = pd.to_datetime(values, errors="raise", utc=True)
            distinct = np.sort(times.unique())
            if len(distinct) < 2:
                raise ValueError("Time split requires at least two distinct timestamps")
            boundary = max(1, min(len(distinct) - 1, int(len(distinct) * (1 - fraction))))
            mask = (times < distinct[boundary]).to_numpy()
            # Equal timestamps never straddle the chronological boundary.
            return rows[mask], rows[~mask]
        raise ValueError("split.kind must be random, stratified, group, or time")

    def identity(self):
        result = {
            "adapter": file_hash(Path(__file__)),
            "public_train": file_hash(self.native.public_dir / self.config.train_file),
        }
        for name in self.sample_paths:
            result["submission_schema/" + name] = file_hash(self.native.public_dir / name)
        return result

    def context(self):
        return {
            "contract": "Read train/test CSVs from /data/public. Fit all preprocessing on train "
            "only. Write the declared prediction artifact with one row per test row in unchanged "
            "order; preserve submission identifiers. Read SCIENTISTTWO_SEED for randomness. "
            "No downloads or external services.",
            "configuration": self.config.model_dump(),
            "features": self.features,
            "problem_type": self.problem_type,
            "columns": list(self.frame.columns),
            "submission_columns": self.submission_columns,
            "prediction_artifacts": [self.prediction_file],
            "training_rows": len(self.frame),
            "sample_training_rows": self.frame.iloc[self.subset[:3]].fillna("").to_dict("records"),
        }

    def baseline(self):
        c = self.config
        estimator = (
            "Ridge(alpha=1.0)"
            if self.problem_type == "regression"
            else ("LogisticRegression(max_iter=1000, random_state=seed)")
        )
        prediction = "model.predict(test)"
        if c.metric == "roc_auc":
            prediction = "model.predict_proba(test)[:, list(model.classes_).index(1)]"
        elif c.metric == "log_loss":
            prediction = (
                "model.predict_proba(test)[:, "
                f"[list(map(str, model.classes_)).index(v) for v in {c.classes!r}]]"
            )
        source = f"""import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

def text_values(values):
    return values.fillna("").astype(str)

seed = int(os.environ.get("SCIENTISTTWO_SEED", "0"))
train = pd.read_csv({("/data/public/" + c.train_file)!r})
test = pd.read_csv({("/data/public/" + c.test_file)!r})
features = {self.features!r}
transforms = []
for index, column in enumerate(features.get("text", [])):
    transforms.append(("text_" + str(index), make_pipeline(
        FunctionTransformer(text_values, validate=False), TfidfVectorizer()), column))
if features.get("numeric"):
    transforms.append(("numeric", make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()),
        features["numeric"]))
if features.get("categorical"):
    transforms.append(("categorical", make_pipeline(
        SimpleImputer(strategy="most_frequent", keep_empty_features=True),
        OneHotEncoder(handle_unknown="ignore")), features["categorical"]))
model = Pipeline([("features", ColumnTransformer(transforms)), ("estimator", {estimator})])
model.fit(train, train[{c.target!r}])
pred = np.asarray({prediction})
out = test.copy()
for index, column in enumerate({c.prediction_columns!r}):
    out[column] = pred[:, index] if pred.ndim == 2 else pred
path = Path({self.prediction_file!r})
path.parent.mkdir(parents=True, exist_ok=True)
out[{self.submission_columns!r}].to_csv(path, index=False)
"""
        return Candidate(
            files=[SourceFile(path="train.py", content=source)],
            command=["python", "train.py"],
            description="Fixed train-fitted tabular preprocessing and linear model reference",
        )

    def prepare(self, phase, destination):
        c = self.config
        public = destination / "public"
        public.mkdir(parents=True, exist_ok=True)
        rows = self.subset if phase == "subset" else self.train
        train = self.frame.iloc[rows]
        test = self.frame.iloc[self.valid].drop(columns=[c.target])
        for name in (c.train_file, c.test_file, *self.sample_paths):
            (public / name).parent.mkdir(parents=True, exist_ok=True)
        train.to_csv(public / c.train_file, index=False)
        test.to_csv(public / c.test_file, index=False)
        sample = test.copy()
        for column in c.prediction_columns:
            sample[column] = 0
        for name in self.sample_paths:
            sample[self.submission_columns].to_csv(public / name, index=False)
        return {
            "public": str(public),
            "files": tree_manifest(public),
            "train_indices": list(map(int, rows)),
            "validation_indices": list(map(int, self.valid)),
            "labels": self.labels[self.valid].tolist(),
            "identifiers": test[self.id_columns].to_dict("list"),
        }

    def validate(self, candidate):
        forbidden = {
            self.config.train_file,
            self.config.test_file,
            self.prediction_file,
            *self.sample_paths,
        }
        if forbidden & {f.path for f in candidate.files}:
            raise ValueError("Candidate cannot supply input data or precomputed submissions")

    def execution(self, prepared, seed):
        return {"kind": "command", "environment": {"SCIENTISTTWO_SEED": str(seed)}}

    def prediction_artifacts(self, prepared):
        return [self.prediction_file]

    def score(self, phase, prepared, workspace, seconds):
        c = self.config
        frame = pd.read_csv(workspace / self.prediction_file)
        labels = np.asarray(prepared["labels"])
        if len(frame) != len(labels) or set(frame) != set(self.submission_columns):
            raise ValueError("Prediction count/columns differ from the development contract")
        for column, expected in prepared.get("identifiers", {}).items():
            if not frame[column].reset_index(drop=True).equals(pd.Series(expected, name=column)):
                raise ValueError("Prediction identifiers differ from test row order")
        pred = frame[c.prediction_columns].to_numpy()
        metrics = list(metric_specs(c))
        result = {name: self._score(name, labels, pred) for name in metrics}
        if not all(np.isfinite(value) for value in result.values()):
            raise ValueError("Nonfinite calculated metric")
        return result if getattr(c, "metric_specs", {}) else float(result[c.metric])

    def _score(self, metric, labels, pred):
        c = self.config
        if metric in {"accuracy", "macro_f1", "f1", "dice"}:
            if pd.isna(pred).any():
                raise ValueError("Missing prediction labels")
            if not set(pred[:, 0]) <= set(self.labels):
                raise ValueError("Prediction labels are outside the declared training classes")
            if metric == "accuracy":
                return float(accuracy_score(labels, pred[:, 0]))
            return float(
                f1_score(
                    labels,
                    pred[:, 0],
                    average="macro" if metric == "macro_f1" else "binary",
                    zero_division=0,
                )
            )
        pred = pred.astype(float)
        if not np.isfinite(pred).all():
            raise ValueError("Nonfinite predictions")
        if metric in {"roc_auc", "log_loss"} and ((pred < 0).any() or (pred > 1).any()):
            raise ValueError("Probabilities outside [0,1]")
        if metric == "log_loss":
            if not np.allclose(pred.sum(axis=1), 1, atol=1e-6):
                raise ValueError("Class probabilities do not sum to one")
            order = sorted(c.classes)
            return float(
                log_loss(
                    labels.astype(str), pred[:, [c.classes.index(x) for x in order]], labels=order
                )
            )
        values = pred[:, 0]
        if metric == "roc_auc":
            return float(roc_auc_score(labels, values))
        if metric in {"rmse", "mse"}:
            value = mean_squared_error(labels, values)
            return float(value**0.5 if metric == "rmse" else value)
        if metric == "mae":
            return float(mean_absolute_error(labels, values))
        if metric == "r2":
            return float(r2_score(labels, values))
        if metric == "smape":
            denominator = np.abs(labels.astype(float)) + np.abs(values)
            ratios = np.divide(
                2 * np.abs(labels - values),
                denominator,
                out=np.zeros_like(values),
                where=denominator != 0,
            )
            return float(100 * np.mean(ratios))
        raise ValueError("Unsupported CSV metric")
