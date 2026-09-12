# Simulation

## Commands and outputs

See the [quick start](../README.md) for installation. `flybird` defaults to
seed 7, 30 seconds, and 30 FPS. `--download` explicitly permits fetching missing
official files. An output directory must be new or empty.

- `--decoder PATH` reuses a frozen decoder instead of fitting one.
- `--evaluate` runs seeds 7, 11, 19, 23, and 41 at 30 seconds.
- `--skip-video` produces simulation and provenance records without images.
- `--duration SECONDS` must align to whole frames. The CLI requires 30 FPS
  because each frame advances the neural model and action decoder.

Each run writes `experiment.json`, `decoder.json`, `provenance.json`, and a
short attribution `README.md`. Video runs also write `demo.mp4`, `demo.png`,
and `selection.json`. Evaluation adds `evaluation.json`.
The video shows fixed percentage conditions on the first evaluation seed;
it does not summarize all seeds. Full outcomes stay in `experiment.json`.

## Model and controls

The intact graph retains every neuron annotated `status == Traced`, including
isolated neurons and VNC neurons. Percentages refer to this retained population,
not publication-level neuron totals. Removal silences both input and output paths.

Integer contact counts are normalized by each postsynaptic neuron's incoming
contact total. Predicted GABA and glutamate transmitters get negative signs;
all other and missing labels get positive signs. This is a model rule, not a
receptor-specific conductance measurement. The import found 3,100 retained
`unclear` labels and 502 retained neurons without a transmitter entry.

An engineered game-state encoder drives a rate reservoir. A fitted action
decoder reads pooled neural activity. Training uses separate seeds 101, 103,
and 107. Frozen decoders bind source hashes, ordered neurons, I/O masks,
reservoir settings, game physics, and neural/action cadence. Incompatible
contracts and overlapping training/evaluation seeds are rejected.

Conditions include 75%, 50%, and 25% retention; visual, central-complex,
MB + DAN proxy, motor-proxy, and VNC removals; node-count-matched random
removals; and a no-recurrence null. Population predicates live in
`src/flybird/data.py`. Masks overlap and are not validated game-function maps.
No recurrence removes effective recurrent dynamics, not anatomical contacts.
Random controls match node count, not degree. No degree-preserving topology
null has been evaluated.

## Verified import

The following MaleCNS v1.0 files were imported on 2026-09-12.
Source: [official downloads](https://male-cns.janelia.org/download/).

Base URL for the filenames below:
`https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`

| Input | Bytes | Rows | SHA-256 |
|---|---:|---:|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14,483,314 | 211,577 | `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` |
| `body-neurotransmitters-male-cns-v1.0.feather` | 43,282,834 | 1,835,518 | `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1,051,241,946 | 151,856,684 | `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1` |

Counting rules:

- Source confidence threshold: 0.5, already applied in the published filenames.
- Retain every annotation with `status == Traced`, of every superclass.
- Retain only positive-weight rows whose **both** endpoints are retained.
- Aggregate duplicate directed body pairs by summing integer contact counts.
- Keep autapses and isolated retained neurons.
- No region, minimum-degree, or stronger contact-count pruning is applied.

Resulting **retained neuron count**: 165,122, including 535 isolated neurons.

Resulting **directed pair-edge count**: 25,563,197.

Resulting **summed anatomical contact count**: 124,025,046.

Excluded annotation rows: 46,455 (including glia, orphans and other non-Traced
statuses). Excluded connection rows: 126,293,487. Excluded contact sum:
187,808,197. The raw source contact sum is 311,833,243; no nonpositive source rows
were observed. These are deliberately separate counting rules, not conflicting
versions of one “synapse count.”

## Recorded evaluation

Historical real-data evaluation on five fixed 30-second courses retained all
75 trials. These measurements are not a fresh evaluation of every code change.
The full records remain in local run storage, not in Git.

| Condition | Pipes passed on seeds 7, 11, 19, 23, 41 |
| --- | --- |
| Intact | 12, 12, 12, 12, 12 |
| 75% neurons retained | 0, 12, 12, 12, 10 |
| 50% or 25% retained | 0, 0, 0, 0, 0 each |
| Visual removal and its random control | 0, 0, 0, 0, 0 each |
| Central complex, MB + DAN proxy, or VNC removal | 12, 12, 12, 12, 12 each |
| Random controls for those three groups | 12, 12, 12, 12, 12 each |
| Motor proxy removal | 0, 0, 0, 0, 0 |
| Motor count-matched random removal | 12, 12, 12, 12, 12 |
| No recurrence | 0, 0, 0, 0, 0 |

Seed 7 is the default video seed, not a summary of the variable 75%-retained
result. The motor comparison depends on the engineered readout. Random controls
match node count, not degree; no degree-preserving topology null was run.
Neither the successful baseline nor these lesions establish biological validity.

## Provenance and limits

Each import/run records dataset version, URLs, SHA-256 hashes, row counts,
filters, retained nodes, directed pair edges, and summed anatomical contacts.
Condition ledgers include removed body IDs, mask hashes, scores, course hashes,
and conserved retained/removed counts. Synthetic counts remain separate.
Implementation and artifact hashes support reproduction, not guaranteed
byte-identical video encoding across platforms.

MaleCNS Consortium / Janelia Research Campus, MaleCNS v1.0,
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
[project](https://male-cns.janelia.org/),
[release notes](https://male-cns.janelia.org/release/).
FlyWire is not used; its public data has a separate CC BY-NC 4.0 restriction.

Anatomy does not provide neural time constants, sensory transduction, a body,
or a task policy. The fly sprite is decorative, not a validated flight model.
Keep failed trials, matched controls, and nonmonotonic outcomes with results.
Claims use the labels verified, repo claim, press claim, and not found;
implementation alone does not verify biological behavior.
