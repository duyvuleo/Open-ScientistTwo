"""Portable task fixtures and unit transport checks; no live model calls.

Fixture agents and transport doubles are explicitly synthetic. Container tests
import the same source/data fixtures but obtain every score from real execution.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scientisttwo.backends import FixtureBackend
from scientisttwo.benchmarks.files import file_hash, tree_manifest
from scientisttwo.benchmarks.models import NativeResources
from scientisttwo.live.adapters import load_development
from scientisttwo.live.models import Candidate, DevelopmentSpec, Hypothesis, SourceFile
from scientisttwo.live.workspace_models import WorkspaceSpec
from scientisttwo.live.workspace_task import WorkspaceRunner, freeze_workspace, preflight, task_spec
from scientisttwo.process import ProcessResult


FINAL_CANARY = "PROTECTED_FINAL_ONLY_WORKSPACE_CANARY_729041"


def numeric_candidate(offset=0.0, *, constant=False):
    source = f"""import json
from pathlib import Path
import numpy as np
import pandas as pd
assert not Path("/data/private").exists()
assert not Path("/evaluator").exists()
train = pd.read_csv("/data/public/train.csv")
test = pd.read_csv("/data/public/test.csv")
constant = {constant!r}
coef = np.polyfit(train["x"], train["target"], 1)
pred = np.full(len(test), train["target"].mean()) if constant else np.polyval(coef, test["x"]) + {offset!r}
pd.DataFrame({{"target": pred}}).to_csv("forecast.csv", index=False)
Path("fit.json").write_text(json.dumps({{"rows": len(train), "linear_component": not constant}}))
"""
    return Candidate(
        files=[SourceFile(path="fit.py", content=source)],
        command=["python", "fit.py"],
        description="Known numerical fixture, train-fitted linear component and calibration",
    )


def classification_candidate(kind="constant", *, text=False):
    if kind == "constant":
        estimator = "pred = np.zeros(len(test), dtype=int)"
    elif kind == "guardrail":
        estimator = 'pred = (test["x"] > train["x"].quantile(0.5)).astype(int).to_numpy()'
    elif text:
        estimator = """from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train["text"], train["target"])
pred = model.predict(test["text"])"""
    else:
        estimator = """from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=2, random_state=0)
model.fit(train[["x"]], train["target"])
pred = model.predict(test[["x"]])"""
    source = f"""from pathlib import Path
import numpy as np
import pandas as pd
assert not Path("/data/private").exists()
assert not Path("/evaluator").exists()
train = pd.read_csv("/data/public/train.csv")
test = pd.read_csv("/data/public/test.csv")
{estimator}
pd.DataFrame({{"target": pred}}).to_csv("forecast.csv", index=False)
"""
    return Candidate(
        files=[SourceFile(path="fit.py", content=source)],
        command=["python", "fit.py"],
        description="Known train-fitted classification fixture",
    )


def make_workspace(root, *, kind="regression", context=None, image="synthetic-image"):
    root = Path(root).resolve()
    directories = {name: root / name for name in ("public", "private", "candidate", "evaluator")}
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-5.0, 5.0, 160)
    held_x = np.linspace(-4.8, 4.8, 31)
    if kind == "regression":
        train = pd.DataFrame({"x": x, "target": 3 * x + 2})
        test = pd.DataFrame({"x": held_x})
        labels = 3 * held_x + 2
        baseline = numeric_candidate(constant=True)
        metric, direction = "calibration_error", "minimize"
        scorer = """error = labels - pred
metrics = {"calibration_error": float(np.sqrt(np.mean(error ** 2))), "worst_error": float(np.max(np.abs(error)))}"""
    else:
        label = (x > 3).astype(int)
        held_label = (held_x > 3).astype(int)
        if kind == "text":
            train = pd.DataFrame(
                {"text": ["sun bright" if v else "moon dark" for v in label], "target": label}
            )
            test = pd.DataFrame({"text": ["sun bright" if v else "moon dark" for v in held_label]})
        else:
            train = pd.DataFrame({"x": x, "target": label})
            test = pd.DataFrame({"x": held_x})
        labels = held_label
        baseline = classification_candidate(text=kind == "text")
        metric, direction = "macro_f1", "maximize"
        scorer = """from sklearn.metrics import accuracy_score, f1_score
metrics = {"macro_f1": float(f1_score(labels, pred, average="macro")), "accuracy": float(accuracy_score(labels, pred))}"""
    train.to_csv(directories["public"] / "train.csv", index=False)
    test.to_csv(directories["public"] / "test.csv", index=False)
    pd.DataFrame({"target": labels}).to_csv(directories["private"] / "labels.csv", index=False)
    (directories["private"] / "marker.txt").write_text(FINAL_CANARY)
    for source in baseline.files:
        (directories["candidate"] / source.path).write_text(source.content)
    evaluator = f"""import json
from pathlib import Path
import numpy as np
import pandas as pd
labels = pd.read_csv("/data/private/labels.csv")["target"].to_numpy()
pred = pd.read_csv("/submission/forecast.csv")["target"].to_numpy()
assert pred.shape == labels.shape and np.isfinite(pred).all()
{scorer}
Path("/grade-output/grade.json").write_text(json.dumps({{"valid_submission": True, "metrics": metrics}}))
"""
    (directories["evaluator"] / "grade.py").write_text(evaluator)
    return freeze_workspace(
        WorkspaceSpec(
            name="fixture-" + kind,
            task_id="portable-" + kind,
            description="Synthetic task data; real execution required",
            public_dir=directories["public"],
            private_dir=directories["private"],
            candidate_dir=directories["candidate"],
            evaluator_dir=directories["evaluator"],
            command=baseline.command,
            evaluator_command=["python", "grade.py"],
            submission="forecast.csv",
            metric=metric,
            direction=direction,
            image=image,
            grader_image=image,
            context=context,
            data_provenance="contract_fixture",
            resources=NativeResources(seconds=30, grading_seconds=30),
        ).model_dump(mode="json")
    )


class CalibrationDevelopment:
    """Trusted test plugin: custom names, numerical regression, multiple artifacts."""

    def __init__(self, native, config):
        self.native, self.config = native, config
        self.frame = pd.read_csv(native.public_dir / "train.csv")
        rows = np.random.default_rng(config.split_seed).permutation(len(self.frame))
        cut = int(len(rows) * 0.8)
        self.train, self.valid = rows[:cut], rows[cut:]

    def identity(self):
        return {
            "fixture_plugin": "calibration-v1",
            "data": file_hash(self.native.public_dir / "train.csv"),
        }

    def context(self):
        return {
            "contract": "Read public training data; fit slope/intercept on training only. Write forecast.csv and fit.json. No novelty claim."
        }

    def baseline(self):
        return numeric_candidate(constant=True)

    def validate(self, candidate):
        if [f.path for f in candidate.files] != ["fit.py"] or candidate.command != [
            "python",
            "fit.py",
        ]:
            raise ValueError("Fixture plugin expects a fit.py source and explicit Python command")

    def prepare(self, phase, destination):
        public = destination / "public"
        public.mkdir(parents=True, exist_ok=True)
        rows = self.train[: len(self.train) // 2] if phase == "subset" else self.train
        self.frame.iloc[rows].to_csv(public / "train.csv", index=False)
        self.frame.iloc[self.valid].drop(columns="target").to_csv(public / "test.csv", index=False)
        return {
            "public": str(public),
            "files": tree_manifest(public),
            "labels": self.frame.iloc[self.valid]["target"].tolist(),
            "training_rows": len(rows),
            "train_indices": rows.tolist(),
            "validation_indices": self.valid.tolist(),
        }

    def execution(self, prepared, seed):
        return {"kind": "command"}

    def prediction_artifacts(self, prepared):
        return ["forecast.csv", "fit.json"]

    def score(self, phase, prepared, workspace, seconds):
        prediction = pd.read_csv(workspace / "forecast.csv")["target"].to_numpy()
        labels = np.asarray(prepared["labels"])
        detail = json.loads((workspace / "fit.json").read_text())
        if detail["rows"] != prepared["training_rows"] or prediction.shape != labels.shape:
            raise ValueError("Output shape or fitted training count differs from declared inputs")
        error = prediction - labels
        return {
            "calibration_error": float(np.sqrt(np.mean(error**2))),
            "worst_error": float(np.max(np.abs(error))),
        }


def install_plugin(monkeypatch, adapter=CalibrationDevelopment, name="fixture-calibration"):
    entry = SimpleNamespace(
        name=name,
        value=f"{__name__}:{adapter.__name__}",
        load=lambda: adapter,
        dist=SimpleNamespace(
            metadata={"Name": "scientisttwo-test-plugin"}, version="0.0.synthetic"
        ),
    )
    monkeypatch.setattr(
        "scientisttwo.live.adapters.entry_points",
        lambda **query: [entry] if query["name"] == name else [],
    )


def development(kind="regression"):
    if kind == "regression":
        return DevelopmentSpec(
            adapter="fixture-calibration",
            metric="calibration_error",
            direction="minimize",
            metric_specs={
                "worst_error": {"direction": "minimize", "hard_min": 0, "max_degradation": 0}
            },
            seeds=[0, 1],
            epsilon=1e-5,
        )
    return DevelopmentSpec(
        adapter="tabular",
        target="target",
        prediction_columns=["target"],
        metric="macro_f1",
        direction="maximize",
        seeds=[0, 1],
        epsilon=1e-5,
        metric_specs={"accuracy": {"direction": "maximize", "max_degradation": 0.01}},
        settings={
            "features": {"text": ["text"]} if kind == "text" else {"numeric": ["x"]},
            "submission_columns": ["target"],
            "prediction_file": "forecast.csv",
        },
    )


class ResearchFixtures(FixtureBackend):
    """Scripted decisions expose branches; all comparison values come from execution."""

    def __init__(self, *, kind="regression", meta_improves=True):
        super().__init__(self.respond)
        self.kind, self.meta_improves = kind, meta_improves
        self.seen, self.refinement_count, self.ablation_count, self.meta_count = [], 0, 0, 0

    def proposal(self, offset=0, constant=False):
        if self.kind == "regression":
            return numeric_candidate(offset, constant=constant)
        return classification_candidate(
            "constant" if constant else "learned", text=self.kind == "text"
        )

    @staticmethod
    def idea(title):
        return Hypothesis(
            title=title,
            hypothesis="Known fixture mechanism",
            novelty_score=0,
            novelty_scope="Synthetic roles, no novelty assessment",
            source_ids=["task"],
        ).model_dump()

    def respond(self, role, inputs):
        assert FINAL_CANARY not in json.dumps(inputs)
        self.seen.append((role, inputs))
        if role in {"limitations", "verify_limitations"}:
            return {
                "limitations": ["Constant reference omits a train-fitted component"],
                "relevant": True,
                "reason": "Synthetic role",
            }
        if role == "seeds":
            return {"ideas": [self.idea(f"seed-{i}") for i in range(inputs["count"])]}
        if role == "evolve":
            assert any(attempt["decision"]["decision"] == "Bad" for attempt in inputs["attempts"])
            return {
                "idea": self.idea("evolved"),
                "change_plan": {
                    "changed_files": ["fit.py"],
                    "reason": "Adjust regression calibration or restore the declared constant classification control",
                },
            }
        if role == "implement":
            title = inputs["idea"]["title"]
            if "parent" in inputs:
                offset = 1.3 if title == "evolved" else float(title.removeprefix("revised-offset-"))
                target = self.proposal(offset, constant=self.kind != "regression")
                parent = inputs["parent"]
                before = next(
                    item["content"] for item in parent["files"] if item["path"] == "fit.py"
                )
                return {
                    "parent_source_ref": parent["parent_source_ref"],
                    "edits": [
                        {
                            "path": "fit.py",
                            "before": before,
                            "after": target.files[0].content,
                            "reason": "Apply the independently declared fixture change to the exact parent",
                        }
                    ],
                    "description": "Prepared numerical fixture through anchored parent edits",
                }
            if title == "seed-1":
                return self.proposal(50, constant=self.kind != "regression").model_dump()
            return self.proposal({"seed-0": 2.0, "seed-2": 1.5, "evolved": 1.3}[title]).model_dump()
        if role == "critic":
            return {
                "decision": "Good",
                "reason": "Synthetic request; controller must veto invalid/non-improving evidence",
            }
        if role == "ablations":
            return {
                "plans": [
                    {
                        "purpose": "Remove the fitted slope/learned text component",
                        "changed_file": "fit.py",
                        "candidate": self.proposal(constant=True).model_dump(),
                    }
                ]
            }
        if role == "rebuttal":
            return {
                "plans": [
                    {
                        "id": "fresh-seed-repeat",
                        "concern_ids": [item["id"] for item in inputs["concerns"]],
                        "purpose": "Repeat incumbent on declared fresh seeds",
                        "seed_count": inputs["action_contract"]["max_seed_count"],
                        "execution_kind": "repeat",
                        "evaluation_split": "development",
                        "diagnostics": [
                            {"kind": "metric", "metric": inputs["action_contract"]["metrics"][0]}
                        ],
                        "changed_file": None,
                        "candidate": inputs["incumbent"]["candidate"],
                    }
                ],
                "unaddressed_concerns": [],
            }
        if role == "ablation_critic":
            self.ablation_count += 1
            return {
                "decision": "Refine" if self.ablation_count == 1 else "Good",
                "reason": "Synthetic calibration refinement request",
            }
        if role == "refine":
            self.refinement_count += 1
            offset = 0.7 if self.refinement_count == 1 else (0.0 if self.meta_improves else 10.0)
            return {
                "idea": self.idea(f"revised-offset-{offset}"),
                "change_plan": {
                    "changed_files": ["fit.py"],
                    "reason": "Measure a declared calibration revision or constant-classifier control",
                },
            }
        if role == "draft":
            evidence = inputs["evidence"]
            return {
                "title": "Fixture evidence draft",
                "claims": [
                    {"experiment_id": e["id"], "metric": name, "value": value}
                    for e in evidence
                    if e["status"] == "success"
                    for name, value in e["metrics"].items()
                ],
                "limitations": [
                    "Synthetic agents and synthetic task data; measured container outputs"
                ],
                "source_ids": ["task"],
                "concern_responses": [
                    {
                        "concern_id": item["concern_id"],
                        "status": {
                            "executed": "executed",
                            "partial": "limited",
                            "unfulfilled": "unaddressed",
                        }[item["status"]],
                        "experiment_ids": item["observation_ids"],
                        "explanation": "Recorded fixture execution coverage; semantic concern resolution remains unverified",
                    }
                    for item in inputs.get("supplementary_coverage", [])
                ],
            }
        if role == "review":
            return {"score": 7, "concerns": ["Synthetic request for a fresh-seed repetition"]}
        if role == "meta":
            self.meta_count += 1
            return {
                "decision": "ReviseMethod" if self.meta_count == 1 else "Accept",
                "reason": "Synthetic method-refinement request",
            }
        raise AssertionError(role)


class WorkspaceTransportDouble:
    """SYNTHETIC transport data only: never used by real-container tests."""

    def __init__(self, *, invalid=False):
        self.calls, self.invalid = [], invalid

    def __call__(self, argv, timeout, **kwargs):
        self.calls.append(argv)
        if "inspect" in argv:
            return ProcessResult(0, "sha256:synthetic-image", "", 0.0)
        if "run" not in argv:
            return ProcessResult(0, "", "", 0.0)
        mounts = {
            argv[i + 1].rsplit(":", 2)[1]: Path(argv[i + 1].rsplit(":", 2)[0])
            for i, arg in enumerate(argv[:-1])
            if arg == "-v"
        }
        assert "--network=none" in argv and "--read-only" in argv
        if "/workspace" in mounts:
            assert "/data/private" not in mounts and "/evaluator" not in mounts
            (mounts["/workspace"] / "forecast.csv").write_text("target\n0\n")
        else:
            assert {"/submission", "/data/private", "/evaluator", "/grade-output"} <= set(mounts)
            grade = {
                "valid_submission": True,
                "metrics": {"calibration_error": "fabricated" if self.invalid else 123.0},
            }
            (mounts["/grade-output"] / "grade.json").write_text(json.dumps(grade))
        return ProcessResult(0, "SYNTHETIC TRANSPORT", "", 0.0)


@pytest.mark.parametrize("kind", ["regression", "classification", "text"])
def test_research_fixture_uses_exact_parent_and_structured_supplementary_contracts(kind):
    """Validate fixture responses without launching a container or measuring scores."""
    from scientisttwo.live.models import Draft
    from scientisttwo.live.parent_edits import MethodChange, ParentPatch
    from scientisttwo.live.supplementary import SupplementaryPlans, negotiate, normalize_concerns

    backend = ResearchFixtures(kind=kind)
    candidate = backend.proposal(2.0)
    parent = {
        "parent_source_ref": "artifacts/" + "a" * 64 + ".json",
        "editable_paths": ["fit.py"],
        "files": [{"path": "fit.py", "content": candidate.files[0].content}],
    }
    for role in ("evolve", "refine"):
        change = MethodChange.model_validate(
            backend.respond(
                role,
                {
                    "parent": parent,
                    "attempts": [{"decision": {"decision": "Bad"}}],
                },
            )
        )
        patch = ParentPatch.model_validate(
            backend.respond(
                "implement",
                {
                    "parent": parent,
                    "idea": change.idea.model_dump(),
                    "change_plan": change.change_plan.model_dump(),
                },
            )
        )
        assert patch.parent_source_ref == parent["parent_source_ref"]
        assert patch.edits[0].before == candidate.files[0].content
        assert patch.edits[0].after != patch.edits[0].before
    concerns = normalize_concerns("review/fixture", ["Repeat on fresh seeds"])
    metric = "calibration_error" if kind == "regression" else "macro_f1"
    requested = SupplementaryPlans.model_validate(
        backend.respond(
            "rebuttal",
            {
                "incumbent": {"candidate": candidate.model_dump()},
                "concerns": [item.model_dump() for item in concerns],
                "action_contract": {"max_seed_count": 2, "metrics": [metric]},
            },
        )
    )
    result = negotiate(requested, concerns, [3, 4], [metric])
    assert result.decisions[0].status == "accepted"
    assert result.decisions[0].required_evaluations == 2
    draft = Draft.model_validate(
        backend.respond(
            "draft",
            {
                "evidence": [],
                "supplementary_coverage": [
                    {"concern_id": concerns[0].id, "status": "unfulfilled", "observation_ids": []}
                ],
            },
        )
    )
    assert draft.claims == []
    assert draft.concern_responses[0].status == "unaddressed"


def test_workspace_freeze_dispatch_and_input_tampering(tmp_path):
    spec = make_workspace(tmp_path / "input")
    assert isinstance(task_spec(spec.model_dump(mode="json")), WorkspaceSpec)
    assert preflight(spec)["ready"]
    (spec.evaluator_dir / "grade.py").write_text("modified evaluator")
    with pytest.raises(ValueError, match="Frozen input tree"):
        preflight(spec)


def test_workspace_overlap_rejected(tmp_path):
    spec = make_workspace(tmp_path / "input")
    with pytest.raises(ValueError, match="separate"):
        WorkspaceSpec.model_validate(
            {**spec.model_dump(), "private_dir": spec.public_dir / "nested"}
        )
    with pytest.raises(ValueError, match="overlap"):
        WorkspaceRunner(spec, spec.public_dir / "output")


def test_plugin_registration_provenance_and_no_final_target_in_context(tmp_path, monkeypatch):
    spec = make_workspace(tmp_path / "input")
    install_plugin(monkeypatch)
    adapter = load_development(spec, development())
    assert adapter.identity()["plugin_distribution"] == "scientisttwo-test-plugin"
    assert adapter.identity()["plugin_source"] == file_hash(Path(__file__))
    assert FINAL_CANARY not in json.dumps(adapter.context())
    prepared = adapter.prepare("full", tmp_path / "development")
    assert not set(prepared["train_indices"]) & set(prepared["validation_indices"])
    assert "target" not in pd.read_csv(Path(prepared["public"]) / "test.csv")


def test_workspace_final_transport_isolated_and_resume_no_repeat(tmp_path):
    spec = make_workspace(tmp_path / "input")
    double = WorkspaceTransportDouble()
    runner = WorkspaceRunner(spec, tmp_path / "run", executor=double)
    try:
        result = runner.run()
        assert result.status == "completed" and result.score == 123.0
        count = len(double.calls)
        assert runner.run() == result and len(double.calls) == count
        assert not result.leaderboard_eligible and result.live_model_calls == 0
    finally:
        runner.close()


def test_workspace_rejects_non_numeric_final_grade(tmp_path):
    spec = make_workspace(tmp_path / "input")
    runner = WorkspaceRunner(
        spec, tmp_path / "run", executor=WorkspaceTransportDouble(invalid=True)
    )
    try:
        with pytest.raises(ValueError, match="nonnumeric"):
            runner.run()
    finally:
        runner.close()


def test_workspace_unknown_execution_is_not_automatically_repeated(tmp_path):
    spec = make_workspace(tmp_path / "input")
    double = WorkspaceTransportDouble()
    runner = WorkspaceRunner(spec, tmp_path / "run", executor=double)
    runner.store.set("workspace_started", True)
    try:
        assert runner.run().status == "interrupted"
        assert not any("run" in call for call in double.calls)
    finally:
        runner.close()


def test_failed_grading_cannot_reuse_previous_grade_file(tmp_path):
    class PartialGrader(WorkspaceTransportDouble):
        def __init__(self):
            super().__init__()
            self.grades = 0

        def __call__(self, argv, timeout, **kwargs):
            grading = any(":/grade-output:" in value for value in argv)
            if not grading:
                return super().__call__(argv, timeout, **kwargs)
            self.grades += 1
            if self.grades == 1:
                super().__call__(argv, timeout, **kwargs)  # Partial JSON before process failure.
                return ProcessResult(1, "SYNTHETIC partial grade", "", 0.0)
            self.calls.append(argv)
            return ProcessResult(0, "SYNTHETIC missing new grade", "", 0.0)

    spec = make_workspace(tmp_path / "input")
    double = PartialGrader()
    runner = WorkspaceRunner(spec, tmp_path / "run", executor=double)
    try:
        assert runner.run().status == "grading_failed"
        assert runner.run().status == "grading_failed"
        assert double.grades == 2
        assert (runner.store.root / "grade-output" / "0" / "grade.json").is_file()
        assert not (runner.store.root / "grade-output" / "1" / "grade.json").exists()
    finally:
        runner.close()
