from types import SimpleNamespace

import pytest

from scientisttwo.benchmarks.runner import NativeRunner


@pytest.mark.parametrize("runtime", ["podman", "docker"])
@pytest.mark.parametrize("prefix", ["", "sha256:"])
def test_inspected_complete_config_digest_is_canonical(runtime, prefix):
    runner = NativeRunner.__new__(NativeRunner)
    runner.prefix = [runtime]
    calls = []

    def inspect(argv, timeout):
        calls.append((argv, timeout))
        return SimpleNamespace(code=0, timed_out=False, stdout=prefix + "abc12345" * 8 + "\n")

    runner.executor = inspect
    assert runner._image("local/image:checked") == "sha256:" + "abc12345" * 8
    assert calls == [([runtime, "image", "inspect", "local/image:checked", "--format", "{{.Id}}"], 15)]


@pytest.mark.parametrize("value", ["sha256:test-fixture", "short-fixture-id", "a" * 63, "a" * 65])
def test_boundary_does_not_guess_missing_digest_bytes(value):
    runner = NativeRunner.__new__(NativeRunner)
    runner.prefix = ["podman"]
    runner.executor = lambda *args: SimpleNamespace(code=0, timed_out=False, stdout=value)
    assert runner._image("fixture") == value


@pytest.mark.parametrize("result", [
    SimpleNamespace(code=1, timed_out=False, stdout="a" * 64),
    SimpleNamespace(code=0, timed_out=True, stdout="a" * 64),
    SimpleNamespace(code=0, timed_out=False, stdout=""),
])
def test_image_inspection_failure_is_not_normalized_into_success(result):
    runner = NativeRunner.__new__(NativeRunner)
    runner.prefix = ["podman"]
    runner.executor = lambda *args: result
    with pytest.raises(ValueError, match="Local image unavailable"):
        runner._image("fixture")
