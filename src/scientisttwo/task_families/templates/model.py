"""Reference architectures. This is the candidate-editable source boundary."""


def build_model(task, config, assets):
    import torch
    from torch import nn

    if task == "cifar10":
        # Matches the upstream LeNet-style architecture; no final-data evaluation.
        hidden = int(config.get("hidden_size", 120))
        return nn.Sequential(
            nn.Conv2d(3, 6, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
            nn.Linear(16 * 5 * 5, hidden), nn.ReLU(), nn.Linear(hidden, 84), nn.ReLU(),
            nn.Linear(84, 10),
        )
    if task == "ogbn-arxiv":
        from torch_geometric.nn import SAGEConv
        hidden = int(config.get("hidden_size", 64))

        class GraphSAGE(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = SAGEConv(config["input_features"], hidden)
                self.second = SAGEConv(hidden, config["classes"])
                self.dropout = float(config.get("dropout", 0.0))

            def forward(self, features, edge_index):
                values = self.first(features, edge_index).relu()
                values = nn.functional.dropout(values, p=self.dropout, training=self.training)
                return self.second(values, edge_index)

        return GraphSAGE()
    if task == "identify-contrails":
        # Explicit audited reference, not an assertion of upstream reproduction.
        hidden = int(config.get("hidden_size", 32))
        return nn.Sequential(nn.Conv2d(config["channels"], hidden, 3, padding=1),
                             nn.ReLU(), nn.Conv2d(hidden, hidden, 3, padding=1),
                             nn.ReLU(), nn.Conv2d(hidden, 1, 1))
    if task == "fathomnet":
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, config["classes"])
        return model
    if task == "imdb":
        from transformers import AutoConfig, AutoModelForSequenceClassification
        architecture = AutoConfig.from_pretrained(assets / "distilbert", local_files_only=True,
                                                   trust_remote_code=False, num_labels=2)
        if "dropout" in config:
            architecture.dropout = config["dropout"]
            architecture.attention_dropout = config["dropout"]
            architecture.seq_classif_dropout = config["dropout"]
        return AutoModelForSequenceClassification.from_pretrained(
            assets / "distilbert", local_files_only=True, config=architecture,
            trust_remote_code=False,
        )
    if task == "babylm":
        from transformers import AutoConfig, AutoModelForCausalLM
        architecture = AutoConfig.from_pretrained(assets / "model_config", local_files_only=True,
                                                  trust_remote_code=False)
        if "dropout" in config:
            architecture.resid_pdrop = architecture.embd_pdrop = architecture.attn_pdrop = config["dropout"]
        return AutoModelForCausalLM.from_config(architecture, trust_remote_code=False)
    if task == "llama-inference":
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            assets / "model", local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float16,
        )
    raise ValueError("Unsupported reference architecture")


def predict_batch(model, task, inputs, *, config, context):
    """Graph candidates receive all public nodes/edges and the requested global IDs."""
    if task == "ogbn-arxiv":
        return model(context["graph_x"], context["edge_index"])[context["node_indices"]]
    return model(inputs)


def optimize_model(model, config):
    """Llama optimization starts from the trusted frozen pretrained model."""
    import torch
    if config.get("compile", False):
        model.forward = torch.compile(model.forward)
    return model
