"""Explicit fixture implementations. These sources are only executed in a container."""

from pydantic import Field

from .models import Idea, Record


def source_for(idea: Idea) -> str:
    parameters = FixtureParameters.model_validate(idea.parameters)
    model = {
        "logistic": f"LogisticRegression(C={parameters.strength!r}, max_iter=2000, random_state=seed)",
        "knn": f"KNeighborsClassifier(n_neighbors={parameters.neighbors}, n_jobs=1)",
        "svc": f"SVC(C={parameters.strength!r}, gamma='scale', random_state=seed)",
        "forest": "RandomForestClassifier(n_estimators=80, max_depth=8, random_state=seed, n_jobs=1)",
    }[idea.method]
    return f"""# Scripted ScientistTwo fixture; not an autonomously discovered method.
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def build(seed):
    estimator = {model}
    return make_pipeline(StandardScaler(), estimator) if {parameters.scaled!r} else estimator
"""


def baseline_idea() -> Idea:
    return Idea(
        id="baseline",
        hypothesis="Fixed standardized logistic regression reference.",
        method="logistic",
        origin="baseline",
    )


class FixtureParameters(Record):
    scaled: bool = True
    strength: float = Field(default=1, gt=0, le=100)
    neighbors: int = Field(default=5, ge=1, le=25)
