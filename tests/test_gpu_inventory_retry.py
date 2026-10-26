"""Synthetic driver observations: retries do not establish live GPU readiness."""

import subprocess
from types import SimpleNamespace

import pytest

import scientisttwo.runtime_profiles as profiles


GPU_A = "GPU-00000000-0000-0000-0000-000000000001"
GPU_B = "GPU-00000000-0000-0000-0000-000000000002"
XML_COMMAND = ["nvidia-smi", "-q", "-x"]
CSV_COMMAND = ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader,nounits"]


def outputs(uuid=GPU_B, minor=4, index=6):
    xml = (
        f"<nvidia_smi_log><gpu><uuid>{uuid}</uuid><minor_number>{minor}</minor_number>"
        "<pci><pci_bus_id>00000000:8D:00.0</pci_bus_id></pci></gpu></nvidia_smi_log>"
    )
    return xml, f"{index}, {uuid}, 00000000:8D:00.0\n"


def sequence(monkeypatch, responses):
    remaining = iter(responses)
    calls = []

    def run(argv, **kwargs):
        assert kwargs == {"check": True, "capture_output": True, "text": True, "timeout": 10}
        calls.append(argv)
        response = next(remaining)
        if response == "timeout":
            raise subprocess.TimeoutExpired(argv, 10, output="PRIVATE DRIVER DUMP")
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(stdout=response)

    monkeypatch.setattr(profiles.subprocess, "run", run)
    return calls


@pytest.mark.parametrize("timed_out_query", ["xml", "csv"])
def test_timeout_retries_a_fresh_complete_pair(monkeypatch, caplog, timed_out_query):
    xml, csv = outputs()
    # The first XML has a different UUID/minor. Reusing it after a CSV timeout
    # must not be accepted as the fresh device mapping.
    first_xml, _ = outputs(GPU_A, minor=7, index=1)
    responses = ["timeout", xml, csv] if timed_out_query == "xml" else [first_xml, "timeout", xml, csv]
    calls = sequence(monkeypatch, responses)

    assert profiles.gpu_inventory() == [
        profiles.GpuDevice(index=6, minor_number=4, uuid=GPU_B, pci_bus_id="00000000:8D:00.0")
    ]
    assert calls == ([XML_COMMAND, XML_COMMAND, CSV_COMMAND] if timed_out_query == "xml"
                     else [XML_COMMAND, CSV_COMMAND, XML_COMMAND, CSV_COMMAND])
    assert "discarding the partial observation" in caplog.text
    assert "recovered after one timeout" in caplog.text
    assert "PRIVATE DRIVER DUMP" not in caplog.text
    assert xml not in caplog.text and first_xml not in caplog.text


@pytest.mark.parametrize("timed_out_query", ["xml", "csv"])
def test_second_timeout_fails_closed_without_a_third_attempt(monkeypatch, caplog, timed_out_query):
    xml, _ = outputs()
    responses = ["timeout", "timeout"] if timed_out_query == "xml" else [xml, "timeout", xml, "timeout"]
    calls = sequence(monkeypatch, responses)
    with pytest.raises(ValueError, match="Cannot verify NVIDIA") as error:
        profiles.gpu_inventory()
    assert isinstance(error.value.__cause__, subprocess.TimeoutExpired)
    assert calls == ([XML_COMMAND, XML_COMMAND] if timed_out_query == "xml"
                     else [XML_COMMAND, CSV_COMMAND, XML_COMMAND, CSV_COMMAND])
    assert caplog.text.count("discarding the partial observation") == 1
    assert "recovered" not in caplog.text
    assert "PRIVATE DRIVER DUMP" not in caplog.text


@pytest.mark.parametrize("query", ["xml", "csv"])
@pytest.mark.parametrize("error", [
    subprocess.CalledProcessError(1, XML_COMMAND, stderr="PRIVATE DRIVER DUMP"),
    OSError("Synthetic driver unavailable"),
])
def test_non_timeout_subprocess_failure_is_never_retried(monkeypatch, caplog, query, error):
    xml, _ = outputs()
    calls = sequence(monkeypatch, [error] if query == "xml" else [xml, error])
    with pytest.raises(ValueError, match="Cannot verify NVIDIA") as caught:
        profiles.gpu_inventory()
    assert caught.value.__cause__ is error
    assert calls == ([XML_COMMAND] if query == "xml" else [XML_COMMAND, CSV_COMMAND])
    assert not caplog.records


@pytest.mark.parametrize("case", ["malformed_xml", "malformed_csv", "pci_mismatch", "unknown_uuid"])
def test_invalid_or_inconsistent_observation_is_never_retried(monkeypatch, caplog, case):
    xml, csv = outputs()
    if case == "malformed_xml":
        xml = "<broken"
    elif case == "malformed_csv":
        csv = "6,missing-field\n"
    elif case == "pci_mismatch":
        csv = csv.replace("8D", "7A")
    else:
        csv = csv.replace(GPU_B, GPU_A)
    calls = sequence(monkeypatch, [xml, csv])
    with pytest.raises(ValueError, match="Cannot verify NVIDIA"):
        profiles.gpu_inventory()
    assert calls == [XML_COMMAND, CSV_COMMAND]
    assert not caplog.records


def test_retry_must_pass_validation_before_reporting_recovery(monkeypatch, caplog):
    xml, csv = outputs()
    calls = sequence(monkeypatch, ["timeout", xml, csv.replace("8D", "7A")])
    with pytest.raises(ValueError, match="PCI identity changed"):
        profiles.gpu_inventory()
    assert calls == [XML_COMMAND, XML_COMMAND, CSV_COMMAND]
    assert "discarding the partial observation" in caplog.text
    assert "recovered" not in caplog.text
