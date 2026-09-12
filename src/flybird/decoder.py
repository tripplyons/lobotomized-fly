"""Portable frozen decoder contract for the default CLI reservoir and I/O."""

from dataclasses import asdict
import hashlib

import numpy as np

from .brain import ReservoirConfig
from .provenance import canonical_json
from .simulation import SimulationConfig


def simulation_contract(config):
    """Training and replay share physics and action cadence, not course length."""
    parameters = asdict(config)
    parameters.pop("duration")
    return parameters


def decoder_contract(graph, simulation_config=None):
    """Bind fitted features to dataset, node order, I/O masks and dynamics."""
    return {
        "version": 2,
        "model": "malecns-engineered-rate-reservoir-v1",
        "dataset_sha256": hashlib.sha256(canonical_json(graph.manifest)).hexdigest(),
        "node_order_sha256": hashlib.sha256(canonical_json(graph.node_ids.tolist())).hexdigest(),
        "parameters": asdict(ReservoirConfig()),
        "simulation": simulation_contract(simulation_config or SimulationConfig()),
        "input_regions": ["visual"], "output_regions": ["motor"],
        "io_masks_sha256": {name: hashlib.sha256(np.packbits(graph.regions[name]).tobytes()).hexdigest()
                            for name in ("visual", "motor")},
    }


def validate_decoder(decoder, graph, evaluation_seeds, simulation_config=None):
    """Reject incompatible, nonfinite, malformed, or train/test-leaking fits."""
    canonical_json(decoder)
    expected = decoder_contract(graph, simulation_config)
    if decoder.get("contract") != expected:
        raise ValueError("Decoder contract does not match graph, I/O, reservoir or simulation parameters")
    training_config = decoder.get("simulation_config")
    if not isinstance(training_config, dict) or set(training_config) != set(asdict(SimulationConfig())):
        raise ValueError("Decoder requires a complete training simulation_config")
    if simulation_contract(SimulationConfig(**training_config)) != expected["simulation"]:
        raise ValueError("Decoder training physics/action cadence does not match replay")
    if decoder.get("reservoir_config") != expected["parameters"]:
        raise ValueError("Decoder training reservoir parameters do not match replay")
    groups = ReservoirConfig().readout_groups
    for name in ("mean", "scale", "weights"):
        values = np.asarray(decoder.get(name), dtype=np.float64)
        if values.shape != (groups,) or not np.isfinite(values).all():
            raise ValueError(f"Decoder {name} must contain {groups} finite values")
        if name == "scale" and np.any(values <= 0):
            raise ValueError("Decoder scales must be positive")
    if type(decoder.get("bias")) not in (int, float) or not np.isfinite(decoder["bias"]):
        raise ValueError("Decoder bias must be a finite scalar")
    seeds = decoder.get("train_seeds")
    if (not isinstance(seeds, list) or not seeds or any(type(s) is not int for s in seeds)
            or len(set(seeds)) != len(seeds) or set(seeds) & set(evaluation_seeds)):
        raise ValueError("Decoder requires unique training seeds disjoint from evaluation seeds")
    for key, expected in (("input_regions", ["visual"]), ("output_regions", ["motor"])):
        if decoder.get(key) != expected:
            raise ValueError("Decoder I/O regions do not match CLI contract")
