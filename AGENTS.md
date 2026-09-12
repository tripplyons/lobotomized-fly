# Repository instructions

## Required reading

- Read [README.md](README.md) before changing the project.
- Read [simulation details](docs/flappy-bird.md) for model, data, and claim changes.
- Read [video generation](docs/arcade-video.md) for rendering and packaging changes.

## Repository rules

- Data provenance: record dataset version, source URLs, SHA-256 hashes, row counts, filters, and retained node, pair-edge, and contact totals for every run or import. Never mix counting rules in one sentence.
- Scientific claims: label engineered inputs, model parameters, encoders, and decoders. Use the curated confidence labels (verified, repo claim, press claim, not found). Keep negative results and matched nulls.
- Licenses: MaleCNS is CC-BY. FlyWire public data is CC BY-NC 4.0. Check `docs/flappy-bird.md` before combining sources, and keep commercial-use limits intact.
- Generated files: do not commit datasets, run outputs, weights, audio, video, or other build artifacts. Keep `runs/`, downloads, and large feather files untracked or ignored.
- Tests: add or update the smallest focused test for simulator, encoder, decoder, null-model, or provenance changes. Run it before broader suites.
- Negative results: keep ablated and rewired controls, failed circuit mappings, and superseded claims in the docs or run record. Do not silently drop them.

## Commits

- Verified integrated changes are committed automatically.
- Stage only the task changes. Preserve unrelated staged and unstaged work.
- Never push without explicit user approval.
