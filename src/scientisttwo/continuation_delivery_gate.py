"""Recognize one audited late-response delivery without changing pinned budgets.

Load this separately hashed operator module outside the pinned scientific
package. The old failed operations stay immutable. Reconciliation consumes a
conservative logical model-operation slot, not another claimed provider call.
Installation authorizes no extension of the original deadline or allowances.
"""
from pathlib import Path
import sqlite3

from . import continuation as core
from . import continuation_delivery as delivery
from .live.runtime import AmbiguousCall, LiveBudgetExhausted
from .store import IntegrityError, canonical, digest, file_digest


def _module_hashes():
    return {"gate": file_digest(Path(__file__)), "continuation": file_digest(Path(core.__file__)),
            "delivery": file_digest(Path(delivery.__file__))}


def _verify_binding(store, binding_ref, *, config=None):
    binding = store.json(binding_ref)
    if (binding.get("schema_version") != "scientisttwo-resolved-delivery-gate-v1"
            or binding.get("store_root") != str(store.root)
            or binding.get("modules") != _module_hashes()):
        raise IntegrityError("Resolved-delivery gate source or store binding changed")
    plan, admission = core._admission(store, binding["plan_ref"], binding["admission_ref"])
    core._verify_parent(store, plan)
    if store.get("delivery-gate/" + plan["continuation_id"]) != binding_ref:
        raise IntegrityError("A different resolved-delivery gate is installed")
    if config is not None and canonical(config) != canonical(store.json(plan["protected_metadata"]["live_spec_ref"])):
        raise IntegrityError("Resolved-delivery gate cannot change campaign configuration")
    checked = delivery.audit_reconciliation(store, binding["receipt_ref"])
    if checked["integrity"] != "pass":
        raise IntegrityError("Late-response reconciliation no longer verifies")
    receipt = store.json(binding["receipt_ref"])
    delivery_plan = store.json(receipt["plan_ref"])
    completed_proof = store.json(delivery_plan["proof_ref"])
    if (receipt["failed_operation"] != plan["replacement_operation"]
            or receipt["failed_operation"] != binding["resolved_failed_operation"]
            or receipt["success_operation"] != binding["recovered_operation"]
            or delivery_plan["failed_record"]["request_hash"] != plan["failed_record"]["request_hash"]
            or receipt["paid_calls"] != 0 or receipt["training_runs"] != 0):
        raise IntegrityError("Reconciliation does not resolve this exact continuation writer delivery")
    if completed_proof["dispatch"]["started_at"] < admission["admitted_at"]:
        raise IntegrityError("The recovered dispatch predates this continuation admission")
    # The failed retry was itself admitted by the unchanged original continuation.
    marker = store.db.execute("SELECT * FROM continuation_operations WHERE key=?",
                              (plan["replacement_operation"],)).fetchone()
    if (marker is None or marker["continuation_id"] != plan["continuation_id"]
            or marker["kind"] != "model" or marker["request_hash"] != plan["failed_record"]["request_hash"]
            or marker["after_writer_completion"] != 0):
        raise IntegrityError("The resolved failure lacks its original continuation admission")
    return binding, plan, admission


def _new_rows(store, plan):
    historical = plan["historical_rows"]["live_operations"]
    return [dict(row) for row in store.db.execute("SELECT * FROM live_operations ORDER BY rowid")
            if row["key"] not in historical]


def audit_delivery_gate(store, binding_ref, *, _allow_final_aggregates=False):
    """Audit the exact resolved exception; all other failures remain blockers."""
    report = {"integrity": "fail", "original_total_token_ceiling_verified": False,
              "historical_unknown_cost": "unchanged_unknown",
              "provider_call_count": "not inferred from logical model-operation count"}
    try:
        binding, plan, admission = _verify_binding(store, binding_ref)
        rows = _new_rows(store, plan)
        recovered = binding["recovered_operation"]
        resolved = binding["resolved_failed_operation"]
        markers = {row["key"]: dict(row) for row in store.db.execute("SELECT * FROM continuation_operations")}
        recovered_seen = False
        pending = []
        for row in rows:
            if row["key"] == recovered:
                # The independent reconciliation audit checked its exact result,
                # cache, request, zero paid calls and once-only token charge.
                recovered_seen = True
                continue
            marker = markers.get(row["key"])
            if (marker is None or marker["continuation_id"] != plan["continuation_id"]
                    or marker["kind"] != row["kind"] or marker["request_hash"] != row["request_hash"]):
                raise IntegrityError("New operation lacks exact continuation admission provenance")
            if row["key"] == resolved:
                if recovered_seen or row["status"] != "failed" or row["tokens"] != 0 or not row["result_ref"]:
                    raise IntegrityError("Resolved delivery failure changed or follows its recovery")
                continue
            if not recovered_seen or marker["after_writer_completion"] != 1:
                raise IntegrityError("A new operation preceded the validated recovered response")
            if row["status"] == "running" or not row["result_ref"]:
                if not _allow_final_aggregates or not core._pending_final_aggregate(store, row):
                    raise IntegrityError("Unresolved new operation blocks the delivery gate")
                pending.append(row["key"])
            if row["kind"] == "model" and row["status"] != "success":
                raise IntegrityError("Another failed model operation remains unresolved")
        if not recovered_seen:
            raise IntegrityError("The recovered completion is missing")
        models = [row for row in rows if row["kind"] == "model"]
        limits = admission["allowance"]
        spec = store.json(plan["protected_metadata"]["live_spec_ref"])
        original_models = [dict(row) for row in store.db.execute("SELECT * FROM live_operations WHERE kind='model'")
                           if row["key"] in plan["historical_rows"]["live_operations"]]
        if len(models) > limits["max_new_calls"] or len(models) + len(original_models) > spec["max_calls"]:
            raise IntegrityError("Original logical model-operation count ceiling exceeded")
        tokens = 0
        prior_tokens = sum(row["tokens"] for row in original_models)
        for row in models:
            if tokens >= limits["token_stop_threshold"] or prior_tokens + tokens >= spec["token_stop_threshold"]:
                raise IntegrityError("A model operation followed an unchanged observed-token stop threshold")
            tokens += row["tokens"]
        report.update(integrity="pass", binding_ref=binding_ref, plan_ref=binding["plan_ref"],
                      admission_ref=binding["admission_ref"], receipt_ref=binding["receipt_ref"],
                      resolved_delivery_failure=resolved, recovered_operation=recovered,
                      new_model_operations=len(models), administrative_model_operations=1,
                      new_observed_tokens=tokens, new_experiments=sum(row["kind"] == "experiment" for row in rows),
                      pending_final_aggregations=pending,
                      deadline=plan["protected_metadata"]["live_deadline"],
                      accounting="Recovery conservatively consumes a logical slot; known tokens charged once; original unknown cost preserved")
    except (IntegrityError, ValueError, OSError, KeyError, TypeError, sqlite3.Error) as error:
        report["error"] = str(error)
    return report


class ResolvedDeliveryBudget:
    """Recognize the exact resolved retry; use the original physical budget."""
    def __init__(self, original, binding_ref):
        if not isinstance(original, core.ContinuationBudget):
            raise IntegrityError("A pinned ContinuationBudget is required")
        self.original, self.store, self.config = original, original.store, original.config
        self.binding_ref = binding_ref
        binding, plan, _ = _verify_binding(self.store, binding_ref, config=self.config)
        if (original.plan_ref != binding["plan_ref"] or original.admission_ref != binding["admission_ref"]):
            raise IntegrityError("Delivery resolution belongs to a different continuation")
        self.plan_ref, self.admission_ref = original.plan_ref, original.admission_ref
        self.plan = plan

    def __getattr__(self, name):
        return getattr(self.original, name)

    def _binding(self):
        return _verify_binding(self.store, self.binding_ref, config=self.config)[0]

    def _mapped(self, key):
        binding = self._binding()
        mapped = self.original._mapped(key)
        return binding["recovered_operation"] if mapped == binding["resolved_failed_operation"] else mapped

    def reserve(self, key, kind, request):
        binding = self._binding()
        mapped = self._mapped(key)
        if mapped == binding["recovered_operation"] and (
                kind != "model" or digest(canonical(request)) != self.plan["failed_record"]["request_hash"]):
            raise IntegrityError("Recovered output can only satisfy the exact original writer request")
        prior = self.store.db.execute("SELECT * FROM live_operations WHERE key=?", (mapped,)).fetchone()
        if prior is not None:
            if mapped not in self.plan["historical_rows"]["live_operations"] and mapped != binding["recovered_operation"]:
                marker = self.store.db.execute("SELECT * FROM continuation_operations WHERE key=?", (mapped,)).fetchone()
                if marker is None or marker["continuation_id"] != self.plan["continuation_id"]:
                    raise IntegrityError("Existing operation has no continuation admission")
            return self.original.delegate.reserve(mapped, kind, request)
        audit = audit_delivery_gate(self.store, self.binding_ref, _allow_final_aggregates=True)
        if audit["integrity"] != "pass":
            raise AmbiguousCall("Resolved-delivery continuation is blocked: " + audit["error"])
        if kind == "model":
            limits = self.original.admission["allowance"]
            if (audit["new_model_operations"] >= limits["max_new_calls"]
                    or audit["new_observed_tokens"] >= limits["token_stop_threshold"]):
                raise LiveBudgetExhausted("Original continuation segment ceiling reached")
        # This is the original Budget/StudyBudget, including unchanged deadline,
        # remaining model counts, experiment count/time, and final-release gate.
        result = self.original.delegate.reserve(mapped, kind, request)
        with self.store.db:
            self.store.db.execute("INSERT INTO continuation_operations VALUES(?,?,?,?,?)", (
                mapped, self.plan["continuation_id"], kind, digest(canonical(request)), 1))
        return result

    def finish(self, key, status, result, seconds, tokens=0):
        binding = self._binding()
        mapped = self._mapped(key)
        if mapped in {binding["resolved_failed_operation"], binding["recovered_operation"]}:
            raise IntegrityError("The resolved failure and recovered completion remain immutable")
        return self.original.finish(mapped, status, result, seconds, tokens)

    def activate_final_grant(self, grant_ref, deadline):
        report = audit_delivery_gate(self.store, self.binding_ref, _allow_final_aggregates=True)
        if report["integrity"] != "pass":
            raise IntegrityError("Resolved-delivery audit must pass before final activation")
        if self.store.get("study_final_release_ref") != grant_ref:
            raise IntegrityError("The original both-arms final release is required")
        key = "delivery-gate/final-audit/" + digest(canonical([self.binding_ref, grant_ref]))
        if prior := self.store.get(key):
            saved = self.store.json(prior)
            if saved.get("binding_ref") != self.binding_ref or saved.get("integrity") != "pass":
                raise IntegrityError("Initial resolved-delivery final audit changed")
        else:
            core._append_metadata(self.store, key, self.store.put_json(report, "resolved-delivery-final-audit"))
        return self.original.delegate.activate_final_grant(grant_ref, deadline)


def install_delivery_gate(controller, plan_ref, admission_ref, receipt_ref):
    """Attach an explicit resolution gate without changing the existing plan."""
    if isinstance(controller.budget, ResolvedDeliveryBudget):
        binding = controller.budget._binding()
        if (binding["plan_ref"], binding["admission_ref"], binding["receipt_ref"]) != (plan_ref, admission_ref, receipt_ref):
            raise IntegrityError("Controller already has a different resolved-delivery gate")
        return controller.budget
    original = core.install_continuation(controller, plan_ref, admission_ref)
    plan, _ = core._admission(controller.store, plan_ref, admission_ref)
    checked = delivery.audit_reconciliation(controller.store, receipt_ref)
    if checked["integrity"] != "pass":
        raise IntegrityError("Delivery receipt failed its independent integrity audit")
    receipt = controller.store.json(receipt_ref)
    if receipt["failed_operation"] != plan["replacement_operation"]:
        raise IntegrityError("The receipt must resolve the exact admitted writer retry")
    binding = {"schema_version": "scientisttwo-resolved-delivery-gate-v1", "store_root": str(controller.store.root),
               "modules": _module_hashes(), "plan_ref": plan_ref, "admission_ref": admission_ref,
               "receipt_ref": receipt_ref, "resolved_failed_operation": receipt["failed_operation"],
               "recovered_operation": receipt["success_operation"], "original_total_token_ceiling_verified": False}
    ref = controller.store.put_json(binding, "resolved-delivery-gate")
    core._append_metadata(controller.store, "delivery-gate/" + plan["continuation_id"], ref)
    budget = ResolvedDeliveryBudget(original, ref)
    checked = audit_delivery_gate(controller.store, ref)
    if checked["integrity"] != "pass":
        raise IntegrityError("Delivery gate installation failed: " + checked["error"])
    controller.budget = controller.runner.budget = budget
    return budget
