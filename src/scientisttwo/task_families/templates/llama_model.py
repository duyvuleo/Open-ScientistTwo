"""Trusted fixed-weight model construction followed by a candidate optimization hook."""
import hashlib
import importlib.util
import json


def state_identity(model):
    import torch
    digest = hashlib.sha256()
    parameters = {}
    for name, tensor in model.state_dict().items():
        while name.startswith("_orig_mod.") or name.startswith("module."):
            name = name.split(".", 1)[1]
        if name in parameters:
            raise ValueError("Ambiguous wrapped state names")
        parameters[name] = tensor
    for name, tensor in sorted(parameters.items()):
        digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode())
        flattened = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
        for offset in range(0, flattened.numel(), 8 * 1024**2):
            digest.update(flattened[offset:offset + 8 * 1024**2].cpu().numpy().tobytes())
    architecture = model.config.to_dict()
    fields = ["model_type", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "vocab_size", "max_position_embeddings", "rope_theta", "rope_scaling",
              "rms_norm_eps", "tie_word_embeddings"]
    structural = {name: architecture.get(name) for name in fields}
    return {"parameters_sha256": digest.hexdigest(), "architecture": structural}


def load_optimized_model(assets, source_directory):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        assets / "model", local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
        attn_implementation="eager", device_map={"": "cuda:0"},
    ).eval()
    before = state_identity(model)
    specification = importlib.util.spec_from_file_location("candidate_optimization", source_directory / "model.py")
    candidate = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(candidate)
    config = json.loads((source_directory / "optimization_config.json").read_text())
    hook = getattr(candidate, "optimize_model", None)
    if not callable(hook):
        raise ValueError("Llama candidates must provide optimize_model(model, config)")
    model = hook(model, config)
    after = state_identity(model)
    if before != after:
        raise ValueError("Optimization altered frozen model parameters or structural architecture")
    return model.eval(), {"before": before, "after": after, "status": "identical_declared_state"}
