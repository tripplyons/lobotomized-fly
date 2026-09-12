# Fly brain

A MaleCNS connectome simulation with a fruit-fly pixel-art video renderer.
Compare an intact model with neuron removals on the same Flappy Bird course.
This is offline simulation and playback, not an interactive game.

## Demo

https://github.com/user-attachments/assets/ae92564b-084c-45d3-a1aa-01af28785328

Selected seed-41 example with 100%, 75%, 72%, and 70% of neurons retained.
Selected for visual variety, not as a representative retention-performance result.
See [video details](docs/arcade-video.md#selected-example) for scores and controls.

## Run

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and FFmpeg on `PATH`.
The first run downloads about 1.11 GB of official MaleCNS data. The full graph
needs several GB of RAM; fitting and simulation can take time.

```sh
uv sync --locked
uv run flybird --download --output runs/demo
```

Open `runs/demo/demo.mp4`. It shows the current fruit-fly artwork with
100%, 75%, 50%, and 25% retained-neuron lanes. All region removals and matched
controls remain in `runs/demo/experiment.json`, even though the video shows
only the four percentage conditions.

```sh
# Evaluate five fixed courses without rendering.
uv run flybird --decoder runs/demo/decoder.json --evaluate --skip-video --output runs/evaluation

# Run tests on small fixtures, without downloading data.
uv run pytest -q
```

- [Simulation](docs/flappy-bird.md): model, data provenance, controls, and limits.
- [Video generation](docs/arcade-video.md): package selected recordings and adjust playback.

## What this models

MaleCNS v1.0 supplies the anatomical graph. Inputs, neural dynamics, action
decoder, and game physics are engineered. Successful play is not evidence of
biological fly behavior or a living brain.

Data: MaleCNS Consortium / Janelia Research Campus,
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
[official downloads](https://male-cns.janelia.org/download/).
No FlyWire data or copied game assets are used. This data license does not
assign a license to this repository's code.

Downloads, fitted decoders, and generated videos stay out of Git.
Keep the complete run directory with any distributed video for attribution,
selection details, and provenance.
