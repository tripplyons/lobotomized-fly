# Video generation

`uv run flybird --output runs/demo` simulates the full comparison and renders
four fixed percentage lanes using the current fruit-fly pixel artwork.
Install FFmpeg on `PATH`. Exports are silent H.264 MP4 at 1920×1080 and 30 FPS,
with a three-second final hold. A 30-second recording gives a 33-second video.

## Package selected recordings

To repackage a run, first extract its percentage episodes into separate files.
Run this after the quick start (use a new trial directory for each experiment):

```sh
uv run python -c '
import json
from pathlib import Path
run = json.loads(Path("runs/demo/experiment.json").read_text())
out = Path("runs/trials")
out.mkdir()
for episode in run["episodes"]:
    if episode["seed"] == run["seeds"][0] and episode["condition"]["kind"] == "amount":
        (out / (episode["condition"]["id"] + ".json")).write_text(json.dumps(episode))
'
```

Then select two to four lanes from the same seed/course:

```sh
uv run python -m flybird.arcade_run \
  --reference runs/demo/experiment.json \
  --trials runs/trials/*.json \
  --select intact amount_75 amount_50 amount_25 \
  --decoder runs/demo/decoder.json \
  --output runs/video --video-duration 30 \
  --rationale "Fixed percentage examples, not a general retention-performance result."
```

The normal simulation produces 75%, 50%, and 25% trials. Custom percentages
require separate simulations with `Condition` and `run_experiment` from the
Python API. The packaging command does not generate trajectories or fit decoders.
Supply **all** explored episode files, not just the displayed subset.

The packager checks source data, decoder, model settings, course geometry,
frame cadence, and anatomical masks. It saves original inputs, all explored
outcomes, selected trials, selection rationale, attribution, and provenance.
Keep that complete bundle with the MP4.

`--video-duration` includes the three-second hold and changes only playback
speed. Without it, playback is 1×. All lanes use the same speed.

## Artwork and playback

- Original 23×23 fruit-fly sprite, pixel lettering, green pipes, and shared sky.
- Live lanes show retained-neuron percentages and recorded scores.
- Dead lanes darken behind a pixel skull, `FLY DIED`, and the final score.
- Sample the last recorded state at or before each video time.
- Freeze each lane at its first dead state, including pipes and score.
- Rotate the dead sprite without adding falling physics or moving the collision.
- Scale world geometry equally on both axes; enlarge artwork with nearest-neighbor sampling.

Sprite size and wing poses do not change the simulation's collision radius.
No trajectories, scores, or crash times are fabricated during rendering.

## Selected example

The latest local export used seed 41 and retained percentages 100, 75, 72, and
70, scoring 12, 10, 6, and 2. It was selected for visual variety, not as a
representative dose-response result: 64% also scored 10. All 14 tested seed-41
percentages and all 33 earlier seed-7 percentages remain in the local run bundle,
including failures and nonmonotonic outcomes. Generated recordings are not
included in a fresh checkout.

See [simulation details](flappy-bird.md) for model limits and data attribution.
