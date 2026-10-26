"""Operator entrypoint boundaries; no SSH, model calls, containers or experiments."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from scientisttwo import continuation_operator as operator
from scientisttwo.cli import app


def mini_package(tmp_path):
    package = tmp_path / "pinned"
    (package / "src/scientisttwo").mkdir(parents=True)
    for name, contents in {"pyproject.toml": "# synthetic package\n", "uv.lock": "# synthetic lock\n",
                           "src/scientisttwo/__init__.py": "SYNTHETIC = True\n"}.items():
        (package / name).write_text(contents)
    values = {p.relative_to(package).as_posix(): {"bytes": p.stat().st_size,
        "sha256": operator.file_hash(p)} for p in package.rglob("*") if p.is_file()}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(values))
    return package, manifest, operator.file_hash(manifest)


def test_verify_package_rejects_changed_extra_and_missing_scientific_source(tmp_path):
    package, manifest, sha = mini_package(tmp_path)
    assert operator.verify_package(package, manifest, sha) == package
    extra = package / "src/scientisttwo/extra.py"
    extra.write_text("# unexpected")
    with pytest.raises(ValueError, match="closure"):
        operator.verify_package(package, manifest, sha)
    extra.unlink()
    (package / "src/scientisttwo/__init__.py").write_text("CHANGED = True")
    with pytest.raises(ValueError, match="Pinned source differs"):
        operator.verify_package(package, manifest, sha)


def test_external_operator_loads_without_modifying_pinned_source(tmp_path):
    package, manifest, sha = mini_package(tmp_path)
    repair = tmp_path / "external_repair.py"
    repair.write_text("from . import SYNTHETIC\nBOUND = SYNTHETIC\n")
    before = (manifest.read_bytes(), (package / "src/scientisttwo/__init__.py").read_bytes())
    source = """
import importlib.util,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('external_operator',sys.argv[1])
op=importlib.util.module_from_spec(spec);spec.loader.exec_module(op)
package=op.verify_package(Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4])
repair=op.load_operator(Path(sys.argv[5]),sys.argv[6],package)
import scientisttwo
print(json.dumps({'bound':repair.BOUND,'science':scientisttwo.__file__,'repair':repair.__file__}))
"""
    result = subprocess.run([sys.executable, "-c", source, operator.__file__, str(package),
        str(manifest), sha, str(repair), operator.file_hash(repair)], capture_output=True, text=True,
        check=True)
    actual = json.loads(result.stdout)
    assert actual["bound"] is True
    assert Path(actual["science"]).is_relative_to(package)
    assert actual["repair"] == str(repair)
    assert before == (manifest.read_bytes(), (package / "src/scientisttwo/__init__.py").read_bytes())


def test_operator_cannot_be_inserted_into_pinned_source_tree(tmp_path):
    package, _, _ = mini_package(tmp_path)
    repair = package / "src/scientisttwo/repair.py"
    repair.write_text("raise AssertionError('Must not execute')")
    with pytest.raises(ValueError, match="outside the pinned package"):
        operator.load_operator(repair, operator.file_hash(repair), package)


def test_cli_forwards_to_fresh_interpreter_with_hashed_operator(monkeypatch):
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return type("Result", (), {"returncode": 0})()
    monkeypatch.setattr(subprocess, "run", run)
    result = CliRunner().invoke(app, ["study", "continuation", "plan", "--package", "/pinned"])
    assert result.exit_code == 0, result.output
    argv = calls[0]
    assert argv[0] == sys.executable and argv[1].endswith("continuation_operator.py")
    module = Path(argv[argv.index("--operator-module") + 1])
    assert argv[argv.index("--operator-sha256") + 1] == hashlib.sha256(module.read_bytes()).hexdigest()
    assert "--allow-live" not in argv


def test_operator_help_does_not_import_runtime_or_construct_backend():
    result = subprocess.run([sys.executable, operator.__file__, "--help"], capture_output=True,
                            text=True, check=True)
    assert "--allow-live" in result.stdout and "{plan,admit,run,audit,finalize}" in result.stdout
