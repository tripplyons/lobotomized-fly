"""Audit and package selected percentage trials without rerunning or editing them.

Usage: python -m flybird.arcade_run --help
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil

from .arcade import ArcadeRenderer
from .decoder import validate_decoder
from .provenance import audit_conditions, write_provenance
from .render import render_video
from .simulation import SimulationConfig, make_course


def assemble(reference, trials, selected_ids, decoder):
    """Check common recording contracts and keep all imported outcomes."""
    baselines = [e for e in reference["episodes"] if e["condition"]["kind"] == "intact"
                 and e["seed"] == trials[0]["seed"]] if trials else []
    if len(baselines) != 1:
        raise ValueError("Require one reference intact episode for the trial seed")
    episodes = baselines + trials
    seed = baselines[0]["seed"]
    config = SimulationConfig(**reference["config"])
    course = make_course(seed, config)
    course_hash = hashlib.sha256(json.dumps(course, sort_keys=True).encode()).hexdigest()
    decoder_hash = hashlib.sha256(json.dumps(decoder, sort_keys=True).encode()).hexdigest()
    definitions = {}
    for episode in episodes:
        condition = episode["condition"]
        if condition["id"] in definitions:
            raise ValueError("Duplicate imported condition")
        definitions[condition["id"]] = condition
        brain = episode["brain"]
        if (episode["seed"] != seed or episode["course_sha256"] != course_hash
                or brain["decoder_sha256"] != decoder_hash
                or brain["dataset_manifest"] != reference["dataset"]
                or brain["parameters"] != decoder["reservoir_config"]
                or brain["input_regions"] != decoder["input_regions"]
                or brain["output_regions"] != decoder["output_regions"]):
            raise ValueError("Imported episode differs in course, decoder, dataset or model contract")
        frames = episode["frames"]
        if len(frames) != round(config.duration * config.fps) + 1:
            raise ValueError("Imported recording has an incompatible duration or cadence")
        for i, frame in enumerate(frames):
            t = i / config.fps
            pipes = [{**p, "x": p["x"] - config.pipe_speed * t} for p in course
                     if -config.pipe_width < p["x"] - config.pipe_speed * t < 1.15]
            if frame["t"] != t or frame["pipes"] != pipes:
                raise ValueError("Recorded course geometry or timestamps differ from configuration")
        terminal = next((f for f in frames if not f["alive"]), frames[-1])
        if terminal["score"] != episode["score"]:
            raise ValueError("Recorded terminal score disagrees with episode result")
    if len(selected_ids) != len(set(selected_ids)) or not set(selected_ids) <= definitions.keys():
        raise ValueError("Select unique imported condition IDs")
    experiment = {"schema_version": 1, "config": asdict(config), "model": reference["model"],
                  "claim": reference["claim"], "dataset": reference["dataset"], "seeds": [seed],
                  "conditions": list(definitions.values()), "episodes": episodes}
    by_id = {e["condition"]["id"]: e for e in episodes}
    selected = dict(experiment, episodes=[by_id[key] for key in selected_ids],
                    conditions=[definitions[key] for key in selected_ids])
    ArcadeRenderer(selected)
    return experiment, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Existing experiment.json with intact trial and config")
    parser.add_argument("--trials", type=Path, nargs="+", required=True, help="All exploratory episode JSON files, not just selected runs")
    parser.add_argument("--select", nargs="+", required=True, help="Two to four condition IDs in screen order")
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("downloads"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rationale", required=True, help="Disclose outcome-based selection and its limits")
    parser.add_argument("--video-duration", type=float, help="Total video seconds including a three-second hold; changes playback speed only")
    args = parser.parse_args()
    if args.video_duration is not None and (not math.isfinite(args.video_duration) or args.video_duration <= 3
            or not math.isclose(args.video_duration * 30, round(args.video_duration * 30), abs_tol=1e-9, rel_tol=0)):
        parser.error("--video-duration must be finite, greater than three seconds and aligned to 30 FPS")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("Output must be a new or empty directory")
    reference = json.loads(args.reference.read_text())
    decoder = json.loads(args.decoder.read_text())
    trials = [json.loads(path.read_text()) for path in args.trials]
    experiment, selected = assemble(reference, trials, args.select, decoder)
    playback_speed = experiment["config"]["duration"] / (args.video_duration - 3) if args.video_duration else 1.
    from .data import load_graph
    print("Loading graph and auditing all imported percentage masks...", flush=True)
    graph = load_graph(args.data)
    if graph.manifest != reference["dataset"]:
        raise ValueError("Reference dataset differs from current import")
    validate_decoder(decoder, graph, experiment["seeds"], SimulationConfig(**experiment["config"]))
    ledger = audit_conditions(graph, experiment)
    experiment["condition_ledger"] = ledger
    selected["condition_ledger"] = [item for item in ledger if item["condition"]["id"] in args.select]
    args.output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    def save(name, value):
        path = args.output / f"{name}.json"
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        artifacts[name] = path
    save("exploration", experiment)
    save("experiment", selected)
    save("decoder", decoder)
    save("selection", {"selected_ids": args.select, "rationale": args.rationale,
                       "outcome_based": True, "general_dose_response_claim": False,
                       "measurements": [{"id": e["condition"]["id"], "score": e["score"],
                                         "survival_time": e["survival_time"]} for e in experiment["episodes"]]})
    sources = args.output / "sources"
    sources.mkdir()
    for i, source in enumerate([args.reference, args.decoder, *args.trials]):
        target = sources / f"{i:02d}-{source.name}"
        shutil.copyfile(source, target)
        artifacts[f"source_{i:02d}"] = target
    record = args.output / "README.md"
    record.write_text("# Recorded percentage comparison\n\n"
        "Keep this complete directory with the video. On-screen percentages are rounded shares of retained model neurons.\n\n"
        "MaleCNS Consortium / Janelia Research Campus, MaleCNS v1.0, CC BY 4.0.\n"
        "https://male-cns.janelia.org/ | https://creativecommons.org/licenses/by/4.0/\n\n"
        "An engineered connectome-constrained model, not biological behavior validation. "
        "Inputs, rate dynamics, transmitter signs, decoder and game physics are engineered or modeled. "
        "No FlyWire data or copied game assets are used. Pixel artwork is newly drawn. "
        "NEURONS labels describe retained model neurons. The fruit-fly illustration stays centered on recorded positions; "
        "its size does not change the collision radius or physics.\n\n"
        + args.rationale + "\n\n"
        "One recorded trial per displayed percentage uses the same course and seed. "
        "No trajectories or crash times were edited. Lanes freeze at their first recorded collision; "
        "this snapshot is quantized to 30 Hz, while survival_time uses 120 Hz physics. "
        f"Playback speed is {playback_speed:.9g}x; recorded simulation times and scores are unchanged. "
        "The last scene holds for three video seconds. The right side of each world is cropped, "
        "with equal horizontal and vertical geometry scales. Fly art is not a collision mask or biological flight validation.\n\n"
        "exploration.json retains all imported percentage outcomes, including failures. "
        "sources/ preserves the original imported files, including any controls present in the reference experiment. "
        "selection.json lists displayed IDs and all measurements. provenance.json records dataset URLs, "
        "hashes, row counts, filters, exact audited node/pair-edge/contact counts, model settings and artifact hashes.\n")
    artifacts["readme"] = record
    print("Rendering one arcade scene...", flush=True)
    rendered = render_video(selected, args.output / "demo.mp4", style="arcade", playback_speed=playback_speed)
    artifacts.update(video=Path(rendered["video"]), poster=Path(rendered["poster"]))
    save("render", rendered)
    write_provenance(args.output / "provenance.json", seed=experiment["seeds"][0],
                     config={"simulation": experiment["config"], "decoder": decoder,
                             "seeds": experiment["seeds"], "conditions": experiment["conditions"],
                             "selected_ids": args.select, "selection_rationale": args.rationale,
                             "render": {"style": "arcade", "fps": 30, "hold_seconds": 3,
                                        "playback_speed": playback_speed, "requested_video_duration": args.video_duration}},
                     synthetic_counts={"nodes": 0, "pair_edges": 0}, dataset_manifest=graph.manifest,
                     artifacts=artifacts, conditions=ledger)
    print(f"Artifacts: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
