"""Tiny manufactured data for official-grader contract tests; never benchmark results."""

import csv
import json
from pathlib import Path


def write_csv(path, columns, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def make_fixture(root, task_id):
    root = Path(root).resolve()
    target = root / task_id
    if target.exists():
        raise FileExistsError("Contract fixtures never overwrite an existing competition directory")
    public, private = target / "prepared/public", target / "prepared/private"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    (target / "CONTRACT_FIXTURE.json").write_text(
        json.dumps({"contract_fixture": True, "not_competition_data": True})
    )
    if task_id == "detecting-insults-in-social-commentary":
        comments = ["quiet moon", "bright sun", "soft moon", "warm sun", "blue moon", "gold sun"]
        write_csv(
            public / "train.csv",
            ["Insult", "Date", "Comment"],
            [(0, "", "training moon"), (1, "", "training sun")],
        )
        write_csv(public / "test.csv", ["Date", "Comment"], [("", c) for c in comments])
        write_csv(
            public / "sample_submission_null.csv",
            ["Insult", "Date", "Comment"],
            [(0, "", c) for c in comments],
        )
        write_csv(
            private / "test.csv",
            ["Insult", "Date", "Comment", "Usage"],
            [(int("sun" in c), "", c, "PublicTest") for c in comments],
        )
        write_csv(
            private / "gold_submission.csv",
            ["Insult", "Date", "Comment"],
            [(int("sun" in c), "", c) for c in comments],
        )
    elif task_id == "spooky-author-identification":
        names = ["EAP", "HPL", "MWS"]
        write_csv(
            public / "train.csv",
            ["id", "text", "author"],
            [(f"train-{i}", f"fixture {name}", name) for i, name in enumerate(names)],
        )
        write_csv(
            public / "test.csv",
            ["id", "text"],
            [(f"test-{i}", f"held-out fixture {name}") for i, name in enumerate(names)],
        )
        write_csv(
            public / "sample_submission.csv",
            ["id", *names],
            [(f"test-{i}", 1 / 3, 1 / 3, 1 / 3) for i in range(3)],
        )
        write_csv(
            private / "test.csv",
            ["id", *names],
            [(f"test-{i}", *[int(i == j) for j in range(3)]) for i in range(3)],
        )
    else:
        raise ValueError("Only the explicitly defined grader contract fixtures are supported")
    return root


def make_candidate(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    (path / "helper.py").write_text(
        'def probability(text):\n    return 0.8 if "sun" in text else 0.2\n'
    )
    (path / "train.py").write_text("""# Contract-test candidate; no competition performance claim.
import csv
import os
from pathlib import Path
from helper import probability
assert not Path('/upstream').exists()
assert not Path('/task/scripts').exists()
assert not Path('/data/detecting-insults-in-social-commentary/prepared/private').exists()
assert not os.environ.get('OPENAI_API_KEY')
with open('/data/public/test.csv') as source:
    rows=list(csv.DictReader(source))
with open('submission.csv','w',newline='') as output:
    writer=csv.DictWriter(output,fieldnames=['Insult','Date','Comment'])
    writer.writeheader()
    writer.writerows({'Insult':probability(r['Comment']),'Date':r['Date'],'Comment':r['Comment']} for r in rows)
""")
    return path
