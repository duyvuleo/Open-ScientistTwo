"""Export a validated task into a live manifest without constructing a backend."""

from pathlib import Path
import re

from ..live.models import LiveSpec
from ..store import Store
from .adapter import PreparedDevelopment
from .preparation import canonical
from .readiness import verify_validation


def pinned_validation_config(validation, receipt):
    """Recover the executed configuration and pin every runtime to recorded IDs.

    The SQLite-referenced specification is authoritative; an edited convenience
    JSON file cannot redefine what an earlier validation established.
    """
    validation = Path(validation).resolve()
    campaign = validation / "campaign"
    if not (campaign / "campaign.sqlite").is_file():
        raise ValueError("Validated campaign evidence is missing")
    store = Store(campaign)
    try:
        original = LiveSpec.model_validate(store.json(store.get("live_spec_ref")))
    finally:
        store.close()
    convenient = LiveSpec.model_validate_json((validation / "validation-spec.json").read_text())
    if convenient.model_dump(mode="json") != original.model_dump(mode="json"):
        raise ValueError("Validation specification file differs from the executed configuration")
    value = original.model_dump(mode="json")
    selectors = {}

    def proven_image(selector, image):
        if not isinstance(image, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None:
            raise ValueError("Validated execution requires a recorded immutable image digest")
        if selector.startswith("sha256:") and selector != image:
            raise ValueError("Declared image digest differs from the validated runtime")
        if selector in selectors and selectors[selector] != image:
            raise ValueError("An image selector resolved differently during validation")
        selectors[selector] = image
        return image

    images = receipt.get("images", {})
    value["native"]["image"] = proven_image(original.native.image, images.get("execution"))
    value["native"]["grader_image"] = proven_image(original.native.grader_image, images.get("grading"))
    declared = value["native"].get("evaluation_pipeline")
    if declared is not None:
        observed = {name: set() for name in ("inference", "score") if declared.get(name) is not None}
        for label, final in receipt.get("finals", {}).items():
            operation = receipt.get("final_operation_keys", {}).get(label, f"final/validation-{label}")
            if operation not in {"final/baseline", "final/selected", f"final/validation-{label}"}:
                raise ValueError("Final image evidence refers to an unrelated operation")
            child_path = campaign / "final" / operation.removeprefix("final/")
            if not (child_path / "campaign.sqlite").is_file() or not final.get("official_report_ref"):
                raise ValueError("Final pipeline image evidence is missing")
            child = Store(child_path)
            try:
                report = child.json(final["official_report_ref"])
            finally:
                child.close()
            actual = report.get("pipeline", {})
            for name in observed:
                stage = actual.get(name)
                if not isinstance(stage, dict):
                    raise ValueError("A declared final stage has no executed image evidence")
                observed[name].add(proven_image(declared[name]["image"], stage.get("image")))
        for name, digests in observed.items():
            if len(digests) != 1:
                raise ValueError("Final stage image identity is absent or inconsistent across validation")
            declared[name]["image"] = next(iter(digests))
    return LiveSpec.model_validate(value)


def check_validated_protocol(native, development, validation, receipt):
    """Agent models/budgets remain independent of this immutable task protocol."""
    expected = pinned_validation_config(validation, receipt)
    if native.model_dump(mode="json") != expected.native.model_dump(mode="json"):
        raise ValueError("Proposed native resources, data, images or evaluator differ from the validated protocol")
    actual_development, expected_development = development.model_dump(mode="json"), expected.development.model_dump(mode="json")
    actual_development["settings"].pop("validation_output", None)
    expected_development["settings"].pop("validation_output", None)
    if actual_development != expected_development:
        raise ValueError("Proposed development evaluation settings or seeds differ from the validated protocol")


def export_live(task_id, root, output, *, model, max_calls, campaign_seconds,
                call_seconds=300, token_stop_threshold=None, agent_output_retries=0,
                supplementary_seed_count=2, literature_network=False, compiler_image=None,
                require_pdf=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    task = root / task_id
    prepared, validation = task / "prepared", task / "validation"
    receipt = verify_validation(prepared, validation)
    if receipt["task_id"] != task_id or receipt.get("actual_task_validated") is not True:
        raise ValueError("Only fully validated actual task inputs can be exported for later live execution")
    if not model.strip() or model.startswith(("scripted-", "fixture", "recorded-", "fake-")):
        raise ValueError("Configure an explicit real agent model; fixture/recorded models are not live targets")
    original = pinned_validation_config(validation, receipt)
    if original.engine_version != "research-v2":
        raise ValueError("New live export requires fresh research-v2 validation; historical receipts are not upgraded")
    value = original.model_dump(mode="json")
    value.update(model=model, max_calls=max_calls, campaign_seconds=campaign_seconds,
                 call_seconds=call_seconds, token_stop_threshold=token_stop_threshold,
                 agent_output_retries=agent_output_retries,
                 supplementary_seed_count=supplementary_seed_count, mode="research")
    value["development"]["settings"]["validation_output"] = str(validation)
    value["model_retries"] = 0
    value["workflow"].update(profile="research-pilot", network_literature=literature_network)
    if literature_network:
        value["workflow"]["literature"] = {"providers": ["local", "arxiv", "crossref"], "max_requests": 64}
    if compiler_image is not None:
        value["workflow"]["compiler_image"] = compiler_image
    if require_pdf:
        value["workflow"]["require_pdf"] = True
    config = LiveSpec.model_validate(value)
    checks = PreparedDevelopment(config.native, config.development).live_readiness()
    if not checks["ready"]:
        raise ValueError("Task is not live-ready: " + "; ".join(checks["errors"]))
    if output.exists() or output.is_relative_to(prepared) or output.is_relative_to(validation):
        raise ValueError("Live manifest destination must be new and outside frozen task/evidence trees")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(canonical(config.model_dump(mode="json")))
    return {"manifest": str(output), "model": model, "readiness": checks,
            "live_model_calls": 0, "live_execution_enabled": False,
            "next_step": "A later explicitly authorized live run still requires --allow-live and its configured budgets"}
