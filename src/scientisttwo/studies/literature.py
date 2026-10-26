"""Small, identical prior-art packet for the two CIFAR study arms.

Bibliographic identities and these short abstract fragments were checked against
the linked arXiv records on 2026-10-23. This packet cannot certify novelty or
predict which permitted training recipe will win.
"""
from ..research.literature import SourceRecord, SourcePassage
from ..store import digest

_RECORDS = (
    ("1512.03385", "Deep Residual Learning for Image Recognition",
     ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"], 2015, "v1",
     "We explicitly reformulate the layers as learning residual functions with reference to the layer inputs"),
    ("1412.6980", "Adam: A Method for Stochastic Optimization",
     ["Diederik P. Kingma", "Jimmy Ba"], 2014, "v9",
     "based on adaptive estimates of lower-order moments."),
    ("1711.05101", "Decoupled Weight Decay Regularization",
     ["Ilya Loshchilov", "Frank Hutter"], 2017, "v3",
     "decoupling the weight decay from the optimization steps taken w.r.t. the loss function."),
)


def study_sources():
    records = []
    for identifier, title, authors, year, version, excerpt in _RECORDS:
        url = "https://arxiv.org/abs/" + identifier
        record = SourceRecord(
            id="arxiv:" + identifier, title=title, authors=authors, year=year,
            arxiv_id=identifier, url=url, provider="local", scholarly=True,
            retrieved_at="2026-10-23T02:51:31+00:00", version=version,
            metadata_status="verified", content_status="abstract",
            passages=[SourcePassage(id="abstract-excerpt", text=excerpt,
                sha256=digest(excerpt.encode()), scope="abstract")],
            content_sha256=digest(excerpt.encode()),
            provenance=[{"provider": "arxiv", "url": url, "status": "identity-and-short-abstract-fragment-checked",
                         "checked_date": "2026-10-23"}],
            limitations=["Short abstract fragment only; no claim about this study's measurements or global novelty."])
        records.append(record.model_dump(mode="json"))
    return records
