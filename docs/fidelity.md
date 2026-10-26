# Relationship to ScientistTwo

This is an independent methodological implementation informed by
[ScientistTwo v1](https://arxiv.org/abs/2609.19644v1). It implements a research
lifecycle for inspection and experimentation. It is not the authors' release or
an empirical reproduction of their reported results.

| Mechanism | Implementation | Boundary or adaptation |
| --- | --- | --- |
| Limitation analysis and hypotheses | Separate structured roles, bounded retries, source records, novelty checks | Supplied or explicitly enabled retrieved evidence; no guarantee of global novelty |
| Baseline and search | Prepared references or declared reconstruction; subset screening, full-development comparison, engineering and evolution | A prepared reference is not a reproduced paper baseline |
| Selection and ablation | Controller-approved candidates, declared controls, independent comparison rules | Admissible control design does not establish semantic component isolation |
| Writing | Canonical offline writer or pinned PaperOrchestra agents | Supplied literature and isolated PDF services adapt the upstream writer |
| Review and rebuttal | Structured concerns, supported supplementary requests, manuscript and method revisions | Local scripted or live reviewers; no exact ScholarPeer integration |
| Meta-review | Accept, manuscript revision, or method revision with bounded continuation | Internal acceptance is not conference acceptance |
| Final evaluation | Freeze the method before protected final evaluation | Added holdout safeguard; final results cannot reopen adaptive research |
| Audit | Artifact hashes, trusted rescoring, selected development re-execution | Identity and execution checks do not prove scientific validity |

The `paper` profile preserves selected reported loop ceilings: 16 limitation
rounds, four search rounds, two engineering refinements, one ablation refinement,
two review/rebuttal rounds, and one meta refinement. Seed scheduling, promotion
thresholds, closing-budget reservations, retry behavior, and final holdout rules
include implementation choices. A profile name alone does not establish exact
paper fidelity.

The paper's model mix, 107-task evaluation, reported gains, review scores, costs,
and human-review findings are **not reproduced here**. The generated quadratic
example tests workflow mechanics using established methods. The 13-slot adapted
testbed is a separate evaluation surface whose full live workflow remains
unverified. See [validation status](validation.md).

## Pinned upstream dependencies

- [PaperOrchestra](https://github.com/google-research/paper-orchestra/tree/ca1b3fa01c2970fc7cda32d16245db38d57b3f56),
  revision `ca1b3fa01c2970fc7cda32d16245db38d57b3f56`: 17 hash-locked upstream
  files are bundled for the adapted writing engine. Frozen literature, bounded
  transport, and isolated PDF tools replace upstream services. The full frontend,
  retrieval system, and plotting pipeline are not included; an external pinned
  checkout is an optional override. See [third-party notices](../THIRD_PARTY_NOTICES.md).
- [MLAgentBench](https://github.com/snap-stanford/MLAgentBench/tree/5d71205cc20a8e95d43aa7cb7120e89ca3323e31),
  revision `5d71205cc20a8e95d43aa7cb7120e89ca3323e31`: benchmark references and
  adapted task preparation/evaluation. Task-specific changes are declared in the
  [recipe catalog](../src/scientisttwo/testbed/catalog.py).

These references establish provenance. Their availability does not establish
that this implementation reproduces either upstream project's results.
