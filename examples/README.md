# Examples

[quickstart.json](quickstart.json) is a generated quadratic regression task for
the default offline research controller. Run it from the repository root using
the [quickstart commands](../README.md#start-with-the-cpu-example).

The task fixes the data seed, split seed, and three estimator seed IDs. It
minimizes mean squared error, also reports mean absolute error, and uses a
declared improvement threshold of `0.0001`. Deterministic estimators can produce
identical values across the three estimator seeds; those values are not three
independent dataset replications.

All agent responses in the default run are scripted. Candidate estimators are
executed in a container and their predictions are scored by the trusted adapter.
The example contains no downloaded dataset, paper, generated manuscript, or
historical run artifact.

[test-adapter](test-adapter) demonstrates the separately installed task-adapter
extension boundary. It is a packaging and protocol fixture.
