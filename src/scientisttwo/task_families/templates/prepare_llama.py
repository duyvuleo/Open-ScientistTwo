"""Build fixed 128-token WikiText workloads and reference tokens in a GPU container.

All model/tokenizer/corpus assets must already exist locally with accepted terms.
This script never downloads, changes model weights, or invokes a research agent.
"""
import argparse
import json
import os
from pathlib import Path


def select_requests(path, tokenizer, split, count=400):
    ids, inputs, masks = [], [], []
    with path.open() as handle:
        for index, text in enumerate(handle):
            if not text.strip():
                continue
            values = tokenizer(text, truncation=True, max_length=128, add_special_tokens=True)
            if len(values["input_ids"]) != 128:
                continue
            ids.append(f"wikitext-{split}-{index}")
            inputs.append(values["input_ids"])
            masks.append(values.get("attention_mask", [1] * 128))
            if len(ids) == count:
                break
    if len(ids) != count:
        raise ValueError("Corpus does not contain the full declared 400-request workload; refusing a silent reduced evaluation")
    return ids, inputs, masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--helpers", type=Path, default=Path("/helpers"))
    parser.add_argument("--assets", type=Path, default=Path("/assets"))
    parser.add_argument("--source", type=Path, default=Path("/source"))
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--variant", default="audited-v1")
    parser.add_argument("--helper-manifest", type=Path)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise ValueError("Trusted Llama reference generation requires an allocated GPU")
    if (args.output / "phases").exists():
        raise ValueError("Refusing to overwrite a generated bundle")
    identity_path = args.assets / "model/scientisttwo-identity.json"
    identity = json.loads(identity_path.read_text())
    model_identity = identity.get("model_identity", "")
    if model_identity != "original-llama-7b" and not model_identity.startswith("adapted-"):
        raise ValueError("Explicit original or adapted licensed model identity is required")
    architecture = json.loads((args.assets / "model/config.json").read_text())
    if architecture.get("hidden_size") != 4096 or architecture.get("num_hidden_layers") != 32:
        raise ValueError("Expected the declared compatible 7B architecture")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.assets / "tokenizer", local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.assets / "model", local_files_only=True,
                                                trust_remote_code=False, torch_dtype=torch.float16,
                                                device_map={"": "cuda:0"}, low_cpu_mem_usage=True,
                                                attn_implementation="eager").eval()
    corpus = args.source / "wikitext-103" if (args.source / "wikitext-103").is_dir() else args.source
    for split, filename, phases in [("development", "wiki.valid.tokens", ["screen", "full"]),
                                     ("final", "wiki.test.tokens", ["final"])]:
        ids, inputs, masks = select_requests(corpus / filename, tokenizer, split)
        input_ids, attention_mask = np.asarray(inputs, dtype=np.int64), np.asarray(masks, dtype=np.int64)
        generated = []
        with torch.inference_mode():
            for offset in range(0, 400, 4):
                batch = {"input_ids": torch.as_tensor(input_ids[offset:offset + 4], device="cuda"),
                         "attention_mask": torch.as_tensor(attention_mask[offset:offset + 4], device="cuda")}
                generated.append(model.generate(**batch, max_new_tokens=1, do_sample=False, pad_token_id=0)[:, -1:].cpu().numpy())
        torch.cuda.synchronize()
        for phase in phases:
            public, private = args.output / "phases" / phase / "public", args.output / "phases" / phase / "private"
            public.mkdir(parents=True)
            private.mkdir(parents=True)
            np.savez(public / "evaluation.npz", ids=np.asarray(ids), input_ids=input_ids, attention_mask=attention_mask)
            np.savez(private / "labels.npz", ids=np.asarray(ids), y=np.concatenate(generated))
    metadata = {
        "dataset_origin": "https://huggingface.co/datasets/Salesforce/wikitext",
        "model_identity": model_identity, "source_variant": "audited-llama7b-fp16-128x1",
        "variant": args.variant,
        "split_policy": "wikitext-103-development-final-document-line-disjoint",
        "generator_seed": args.seed, "requests_per_partition": 400, "context_length": 128,
        "generation_length": 1, "batch_size": 4, "dtype": "float16",
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        "deviations": ["Explicit FP16 reference instead of upstream default FP32", "Independent final token equivalence and paired timing replace absent upstream grader"],
    }
    (args.output / "split_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"status": "generated", "task": "llama-inference", "variant": metadata["source_variant"], "live_model_calls": 0}))


if __name__ == "__main__":
    main()
