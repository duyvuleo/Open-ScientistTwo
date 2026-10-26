# Third-party notices

The top-level MIT license covers the original Open-ScientistTwo implementation.
It does not replace the licenses or notices of the third-party assets below.
Original copyright and attribution headers are retained.

## Bundled LaTeX template assets

`src/scientisttwo/research/templates/iclr2025/` contains unmodified template assets
copied from [PaperOrchestra](https://github.com/google-research/paper-orchestra)
at revision `ca1b3fa01c2970fc7cda32d16245db38d57b3f56`. The adjacent
`PROVENANCE.json` records the original file hashes.

- `iclr2025_conference.sty` retains its ICLR/NIPS adaptation credit, including
  Hugo Larochelle's credit. The upstream Apache-2.0 license is retained in the
  template directory's `LICENSE`.
- `iclr2025_conference.bst` retains notices for Patrick W. Daly, Hal Daumé III,
  and the modification by J. Fürnkranz. Its header explicitly permits
  redistribution/modification under the LaTeX Project Public License, version 1
  or later. A verbatim copy of [LPPL 1.3c](https://www.latex-project.org/lppl/lppl-1-3c.txt)
  is included at `licenses/LPPL-1.3c.txt`. The file-specific notice is not replaced
  by the adjacent Apache license or the project's MIT license.

The source templates remain unchanged. Any runtime compatibility changes are
made to a separately staged copy and recorded in that run's provenance.
Conference-style headers are template formatting; they do not indicate submission
to, acceptance by, or affiliation with a conference.

## Bundled PaperOrchestra engine

`src/scientisttwo/_vendor/paperorchestra/` contains the 17 unchanged files in
the adapter's hash-locked execution/template closure from the PaperOrchestra
revision named above. Its Apache-2.0 `LICENSE`, copyright headers and provenance
are included. The bundle excludes the upstream frontend, provider SDK clients,
retrieval clients, plotting pipeline, examples, datasets and Git history. The
Open-ScientistTwo adapter replaces model and PDF services and supplies frozen
literature; these adaptations do not modify the vendored source files.

The bundled `fancyhdr.sty` credits Piet van Oostrum and declares LPPL version 1
or later. `natbib.sty` credits Patrick W. Daly and declares LPPL version 1 or
later. Its notice requires the original `natbib.dtx`; the matching
`2009/07/16 8.31` source is included under the bundle's `license_sources/`, with
its archive URL, version and SHA-256 recorded separately. These file-specific
notices remain applicable. LPPL 1.3c is included in `licenses/LPPL-1.3c.txt`.

An external PaperOrchestra checkout may optionally override the bundle, but it
must satisfy the same pinned source hashes.

## Separately obtained benchmark integrations

[MLAgentBench](https://github.com/snap-stanford/MLAgentBench) and
[MLE-bench](https://github.com/openai/mle-bench) checkouts, datasets, model weights,
and generated benchmark workspaces are not included in this repository. Their
code licenses, data licenses and access terms apply when users obtain them.
Preparation may copy declared upstream utilities into a local generated task
bundle with recorded provenance; this does not grant additional redistribution
rights to third-party data or code.

Python dependencies and container distributions are installed separately using
the included dependency specifications and retain their respective licenses.

## Research attribution

This is an independent implementation of the workflow described in
[ScientistTwo](https://arxiv.org/abs/2609.19644). It is not the authors' released
implementation and does not claim to reproduce their reported benchmark or
review scores. See `docs/fidelity.md` and `CITATION.cff`.
