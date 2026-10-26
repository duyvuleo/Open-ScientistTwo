from typer.testing import CliRunner
import pytest

from scientisttwo.cli import app
from scientisttwo.testbed import cli


@pytest.mark.parametrize("level,folder", [("smoke", "validation-smoke"), ("full", "validation")])
def test_smoke_and_full_keep_separate_resume_checkpoints(tmp_path, monkeypatch, level, folder):
    seen = []
    def validate(prepared, output, **kwargs):
        seen.append((prepared, output, kwargs))
        return {"gates": {"synthetic_test": True}}
    monkeypatch.setattr(cli, "validate", validate)
    result = CliRunner().invoke(app, ["benchmark", "validate", "vectorization", "--root", str(tmp_path),
                                     "--image", "fixture:test", "--level", level])
    assert result.exit_code == 0, result.output
    assert seen[0][1] == tmp_path / "vectorization" / folder
    assert seen[0][2]["max_prompt_chars"] == 200_000


def test_validation_explicit_prompt_ceiling_is_forwarded(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "validate", lambda *args, **kwargs:
        seen.append(kwargs) or {"gates": {"synthetic_test": True}})
    result = CliRunner().invoke(app, ["benchmark", "validate", "vectorization", "--root", str(tmp_path),
        "--image", "fixture:test", "--level", "full", "--max-prompt-chars", "600000"])
    assert result.exit_code == 0, result.output
    assert seen[0]["max_prompt_chars"] == 600_000


def test_validation_rejects_too_small_prompt_ceiling_before_execution(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid prompt limit reached execution")
    monkeypatch.setattr(cli, "validate", forbidden)
    result = CliRunner().invoke(app, ["benchmark", "validate", "vectorization", "--root", str(tmp_path),
        "--image", "fixture:test", "--max-prompt-chars", "999"])
    assert result.exit_code == 2
    assert "1000" in result.output
