# Architecture

Open-ScientistTwo separates scientific proposals from execution and evaluation.
The shared `ResearchController` persists stage decisions before moving through
limitation analysis, hypothesis search, experiments, refinement, writing, review,
and final evaluation. Scripted and live agents use the same structured contracts.

```text
Task + frozen policy + source evidence
                 |
        ResearchController <----> Agent backend
                 |                  scripted / live
            Task adapter
                 |
         Isolated experiment
                 |
     Trusted scoring + comparison
                 |
      Frozen development evidence
                 |
       Writer + review / rebuttal
                 |
       Method freeze -> final evaluation
                 |
          Sealed report bundle
```

## Execution and evidence

The task adapter defines candidate-visible inputs, data partitions, metrics,
controls, and promotion rules. Candidate source is data until the runner executes
it inside a constrained container. Trusted scoring consumes observations and
checks the declared protocol; candidate-reported scores do not decide promotion.
Installed adapters and evaluator code are part of the trusted computing base.

The store combines SQLite state with content-addressed artifacts. Operation
records bind requests, source identities, results, and usage. Resume reuses
committed operations and rejects incompatible source or protocol changes.
An ambiguous live completion is not assumed to have consumed zero tokens and is
not automatically safe to replay.

Development results can influence hypothesis selection and revisions. Final
evaluation occurs after method freeze. The writer receives frozen development
evidence, while final results enter a deterministic appendix after the adaptive
loops. An audit can check artifact integrity, recompute scores, and re-execute
selected development code; these checks do not prove that a scientific claim is
novel or that every implementation matches its description.

## Writers and review

The default canonical writer uses scripted responses for the offline example.
The optional [PaperOrchestra integration](paperorchestra.md) executes the bundled,
pinned upstream writing agents through bounded services. It binds the authoritative
LaTeX and PDF to an evidence index, preserving the previous manuscript across
revisions. Writing calls and PDF operations share campaign accounting.

Reviewers may request supported supplementary experiments or revisions.
Manuscript-only revisions are routed separately from method revisions. A method
change must pass the applicable experiments and comparisons before downstream
writing continues. Internal acceptance is an implementation state, not a
scientific certification.

## Extension points

The bundled TaskSpec path exposes the `TaskAdapter` protocol in
[`tasks.py`](../src/scientisttwo/tasks.py). A separately installed trusted package
can register an adapter under the `scientisttwo.tasks` entry-point group; see
[`examples/test-adapter`](../examples/test-adapter). Native development adapters
use the `scientisttwo.native_development` group. Their schemas, preparation
contracts, and graders are separate from the small CPU demonstration.

Key source modules are [`research/controller.py`](../src/scientisttwo/research/controller.py),
[`research/policy.py`](../src/scientisttwo/research/policy.py),
[`store.py`](../src/scientisttwo/store.py), and
[`runner.py`](../src/scientisttwo/runner.py).
