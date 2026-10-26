# Historical frozen-evidence fixture

The `materials` directory contains an aggregate development-evidence export
prepared on 23 October 2026 from an earlier scripted-agent CPU campaign. It is
retained for the evidence-import contract test and the optional
[standalone writing example](../../docs/paperorchestra.md#write-from-already-frozen-evidence).
It is not a result from the current public release.

The source used the bundled Iris, Wine, and Digits classification tasks and
recorded no approved improvement. Measurements came from CPU experiments;
research decisions were scripted. The manifest binds seven payload files and
identifies the upstream PaperOrchestra revision. It includes no final-test
evidence, datasets, predictions, labels, model binaries, or credentials.

Original observation references and hashes retain provenance, but their raw
campaign payloads are not included. Importing this bundle verifies its file
identities and structure; it does not independently recompute its historical
metrics or establish citation entailment. No historical PDF is distributed.

For a new experiment, use [quickstart.json](../quickstart.json) and report that
run's own status, audit, and resume results.
