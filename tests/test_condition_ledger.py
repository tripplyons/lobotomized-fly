from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import sys

import numpy as np
import pytest
from scipy import sparse

from flybird.brain import Condition, ConnectomeBrain
from flybird.data import load_graph
from flybird.provenance import audit_conditions, build_provenance
from test_data import fixture_data


def small_run(graph):
    conditions = [Condition("intact", "Intact"), Condition("half", "Half", "amount", .5),
                  Condition("visual", "Visual", "region", 1, "visual"),
                  Condition("null", "No recurrence", "no_recurrence")]
    episodes = [{"condition": asdict(c), "seed": s, "brain": ConnectomeBrain(graph, c, s).manifest(),
                 "score": 0, "course_sha256": hashlib.sha256(str(s).encode()).hexdigest()}
                for s in (7, 11) for c in conditions]
    return {"seeds": [7, 11], "conditions": list(map(asdict, conditions)), "episodes": episodes}


def record(graph, run, ledger):
    return build_provenance(seed=7, config={k: run[k] for k in ("seeds", "conditions")},
                            synthetic_counts={"nodes": 0, "pair_edges": 0},
                            conditions=ledger, dataset_manifest=graph.manifest, source_paths=[])


def test_exact_contacts_and_no_recurrence(fixture_data):
    graph = load_graph(fixture_data)
    # Beyond float64 exact integer range; provenance must not use signed weights.
    graph.contacts.data[0] = 2**53 + 1
    graph.manifest["anatomical_counts"]["summed_contacts"] = sum(map(int, graph.contacts.data))
    run = small_run(graph)
    ledger = audit_conditions(graph, run)
    assert len(record(graph, run, ledger)["conditions"]) == 8
    assert ledger[0]["retained"]["contacts"] == 2**53 + 10
    assert ledger[2]["removed_ids"] == [10]
    assert ledger[2]["retained"]["contacts"] == 0
    assert ledger[3]["retained"] == ledger[0]["retained"]
    assert ledger[3]["effective_recurrent_contacts"] == 0
    for item in ledger:
        for key in ("nodes", "pair_edges", "contacts"):
            assert item["retained"][key] + item["removed"][key] == item["total"][key]


@pytest.mark.parametrize("fault", ["missing", "duplicate", "counts", "mask", "ids", "null", "arbitrary"])
def test_incomplete_or_corrupt_real_ledgers_rejected(fixture_data, fault):
    graph = load_graph(fixture_data)
    run = small_run(graph)
    ledger = audit_conditions(graph, run)
    if fault == "missing": ledger.pop()
    if fault == "duplicate": ledger.append(deepcopy(ledger[0]))
    if fault == "counts": ledger[0]["retained"]["contacts"] += 1
    if fault == "mask": ledger[0].pop("mask_sha256")
    if fault == "ids": ledger[2]["removed_ids"] = [999]
    if fault == "null": ledger[3]["effective_recurrent_contacts"] = 10
    if fault == "arbitrary": ledger = [{"name": "good", "score": 99}]
    with pytest.raises(ValueError): record(graph, run, ledger)


def test_audit_checks_actual_mask_and_raw_graph(fixture_data):
    graph = load_graph(fixture_data)
    run = small_run(graph)
    run["episodes"][0]["brain"]["removed_ids"] = [10]
    with pytest.raises(ValueError, match="lesion"): audit_conditions(graph, run)
    graph.contacts = sparse.csr_matrix(graph.contacts, dtype=np.float64)
    with pytest.raises(ValueError, match="int64"): audit_conditions(graph, small_run(graph))


def test_cli_small_real_graph_and_frozen_decoder(fixture_data, tmp_path, monkeypatch):
    from flybird import cli, data, simulation
    from flybird.decoder import validate_decoder
    graph = load_graph(fixture_data)
    monkeypatch.setattr(data, "load_graph", lambda *a, **kw: graph)
    train = simulation.train_connectome_decoder
    monkeypatch.setattr(simulation, "train_connectome_decoder", lambda g: train(g, duration=.2))
    def invoke(path, *extra):
        monkeypatch.setattr(sys, "argv", ["flybird", "--output", str(path), "--duration", ".2", "--skip-video", *extra])
        cli.main()
        return json.loads((path / "provenance.json").read_text())
    first = invoke(tmp_path / "first")
    assert len(first["conditions"]) == 15
    assert first["conditions"][-1]["effective_recurrent_contacts"] == 0
    def forbidden(*a, **kw): raise AssertionError("Frozen decoder must not train")
    monkeypatch.setattr(simulation, "train_connectome_decoder", forbidden)
    second = invoke(tmp_path / "second", "--decoder", str(tmp_path / "first/decoder.json"))
    assert first["conditions"] == second["conditions"]
    assert first["artifacts"]["experiment"]["sha256"] == second["artifacts"]["experiment"]["sha256"]
    decoder = json.loads((tmp_path / "first/decoder.json").read_text())
    validate_decoder(decoder, graph, [7])
    with pytest.raises(ValueError, match="training seeds"): validate_decoder(decoder, graph, [101])
    for field, value in (("scale", [0] * 128), ("weights", [1]), ("contract", {})):
        broken = deepcopy(decoder)
        broken[field] = value
        with pytest.raises(ValueError): validate_decoder(broken, graph, [7])
    # Course length and held-out seed can differ, but neural cadence/physics cannot.
    validate_decoder(decoder, graph, [11], simulation.SimulationConfig(duration=2))
    for changes in ({"fps": 60}, {"gravity": 1.4}, {"physics_hz": 240},
                    {"flap_cooldown": .2}, {"gap_size": .4}):
        with pytest.raises(ValueError, match="simulation"):
            validate_decoder(decoder, graph, [7], simulation.SimulationConfig(**changes))
        broken = deepcopy(decoder)
        broken["simulation_config"].update(changes)
        with pytest.raises(ValueError, match="physics/action cadence"):
            validate_decoder(broken, graph, [7])
    broken = deepcopy(decoder)
    broken["contract"]["version"] = 1
    with pytest.raises(ValueError, match="contract"):
        validate_decoder(broken, graph, [7])
