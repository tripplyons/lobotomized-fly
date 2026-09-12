"""Produce an offline MaleCNS Flappy Bird experiment and comparison video."""

import argparse
import json
import math
from pathlib import Path

from .evaluation import EVALUATION_SEEDS, evaluate_baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("downloads"))
    parser.add_argument("--download", action="store_true", help="Fetch missing official data, about 1.1 GB")
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--decoder", type=Path, help="Reuse a frozen decoder.json with a matching graph/I/O/dynamics contract")
    parser.add_argument("--evaluate", action="store_true", help="Run all five fixed evaluation seeds")
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0 or args.fps <= 0:
        parser.error("duration and fps must be positive and finite")
    if args.fps != 30:
        parser.error("--fps must be 30: the frozen decoder requires 30 Hz neural/action cadence")
    from .simulation import SimulationConfig
    try:
        simulation_config = SimulationConfig(duration=args.duration, fps=args.fps)
    except ValueError as error:
        parser.error(str(error))
    if args.evaluate and args.duration != 30:
        parser.error("--evaluate requires --duration 30")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error(f"Output must be a new or empty directory: {args.output}")

    from .brain import ConnectomeBrain, connectome_conditions
    from .data import load_graph
    from .provenance import audit_conditions, write_provenance
    from .decoder import decoder_contract, validate_decoder
    from .simulation import run_experiment, train_connectome_decoder

    args.output.mkdir(parents=True, exist_ok=True)
    artifacts = {}

    def save(name, value):
        path = args.output / f"{name}.json"
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        artifacts[name] = path

    print("Loading the traced MaleCNS graph...", flush=True)
    graph = load_graph(args.data, download=args.download)
    seeds = list(EVALUATION_SEEDS) if args.evaluate else [args.seed]
    if args.decoder:
        print("Loading frozen engineered decoder...", flush=True)
        decoder = json.loads(args.decoder.read_text())
    else:
        if set(seeds) & {101, 103, 107}:
            parser.error("Evaluation seed overlaps decoder training seeds (101, 103, 107)")
        print("Fitting the engineered decoder on separate training courses...", flush=True)
        decoder = train_connectome_decoder(graph)
        decoder["contract"] = decoder_contract(graph, simulation_config)
    validate_decoder(decoder, graph, seeds, simulation_config)
    save("decoder", decoder)

    def controller(condition, seed):
        return ConnectomeBrain(graph, condition, seed, decoder=decoder)

    print("Running intact, removal, and matched-control trials...", flush=True)
    experiment = run_experiment(
        seed=args.seed, duration=args.duration, fps=args.fps,
        seeds=seeds,
        conditions=connectome_conditions(["visual", "central_complex", "mushroom_body", "motor", "vnc"]),
        controller_factory=controller,
    )
    experiment["dataset"] = graph.manifest
    ledger = audit_conditions(graph, experiment)
    experiment["condition_ledger"] = ledger
    save("experiment", experiment)
    if args.evaluate:
        save("evaluation", evaluate_baseline(experiment))
    if not args.skip_video:
        from .render import render_video

        print("Rendering the fruit-fly video...", flush=True)
        displayed = [episode for episode in experiment["episodes"]
                     if episode["seed"] == seeds[0]
                     and episode["condition"]["kind"] in {"intact", "amount"}]
        selected = dict(experiment, episodes=displayed,
                        conditions=[episode["condition"] for episode in displayed])
        save("selection", {"seed": seeds[0],
                           "condition_ids": [episode["condition"]["id"] for episode in displayed],
                           "outcome_based": False,
                           "rationale": "Fixed intact and percentage conditions on the first evaluation seed. "
                                        "All conditions and seeds remain in experiment.json."})
        rendered = render_video(selected, args.output / "demo.mp4")
        artifacts.update(video=Path(rendered["video"]), poster=Path(rendered["poster"]))
    (args.output / "README.md").write_text(
        "# MaleCNS fruit-fly simulation\n\n"
        "Keep this complete directory with the video. See experiment.json for all controls and outcomes.\n\n"
        "MaleCNS Consortium / Janelia Research Campus, MaleCNS v1.0, CC BY 4.0.\n"
        "https://male-cns.janelia.org/ | https://creativecommons.org/licenses/by/4.0/\n\n"
        "Inputs, neural dynamics, decoder and game physics are engineered or modeled. "
        "This is not biological behavior validation. No FlyWire data or copied game assets are used.\n"
    )
    artifacts["readme"] = args.output / "README.md"
    write_provenance(
        args.output / "provenance.json", seed=args.seed,
        config={"simulation": experiment["config"], "decoder": decoder,
                "seeds": experiment["seeds"], "conditions": experiment["conditions"]},
        synthetic_counts={"nodes": 0, "pair_edges": 0}, dataset_manifest=graph.manifest,
        artifacts=artifacts, conditions=ledger,
    )
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
