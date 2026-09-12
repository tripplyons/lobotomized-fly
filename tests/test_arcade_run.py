from copy import deepcopy
import json
import shutil
import sys

import pytest

from flybird import arcade_run, data
from flybird.brain import Condition, ConnectomeBrain
from flybird.decoder import decoder_contract
from flybird.provenance import sha256_file
from flybird.simulation import SimulationConfig, run_experiment, train_connectome_decoder
from test_data import fixture_data


@pytest.fixture
def recorded(fixture_data):
    graph = data.load_graph(fixture_data)
    config = SimulationConfig(duration=.2)
    decoder = train_connectome_decoder(graph, duration=.2)
    decoder["contract"] = decoder_contract(graph, config)
    conditions = [Condition("intact", "Intact"), Condition("half", "Half", "amount", .5),
                  Condition("quarter", "Quarter", "amount", .25)]
    reference = run_experiment(config=config, conditions=conditions,
        controller_factory=lambda c, s: ConnectomeBrain(graph, c, s, decoder=decoder))
    reference["dataset"] = graph.manifest
    return graph, reference, decoder


def test_assembly_keeps_unselected_trials_and_original_frames(recorded):
    _, reference, decoder = recorded
    before = deepcopy(reference)
    all_trials, selected = arcade_run.assemble(reference, reference["episodes"][1:], ["intact", "half"], decoder)
    assert len(all_trials["episodes"]) == 3
    assert len(selected["episodes"]) == 2
    assert selected["episodes"][1] is reference["episodes"][1]
    assert reference == before


@pytest.mark.parametrize("fault", ["seed", "decoder", "geometry", "cadence", "score", "duplicate", "selection"])
def test_assembly_rejects_incompatible_recordings(recorded, fault):
    _, reference, decoder = recorded
    trials = deepcopy(reference["episodes"][1:])
    selection = ["intact", "half"]
    if fault == "seed": trials[1]["seed"] = 8
    if fault == "decoder": trials[0]["brain"]["decoder_sha256"] = "a" * 64
    if fault == "geometry": trials[0]["frames"][0]["pipes"][0]["x"] += .1
    if fault == "cadence": trials[0]["frames"].pop()
    if fault == "score": trials[0]["score"] = 99
    if fault == "duplicate": trials.append(trials[0])
    if fault == "selection": selection.append("missing")
    with pytest.raises(ValueError):
        arcade_run.assemble(reference, trials, selection, decoder)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_packaged_cli_hashes_sources_and_audits_all_outcomes(recorded, tmp_path, monkeypatch):
    graph, reference, decoder = recorded
    monkeypatch.setattr(data, "load_graph", lambda *a, **kw: graph)
    values = {"reference": reference, "decoder": decoder,
              "half": reference["episodes"][1], "quarter": reference["episodes"][2]}
    for name, value in values.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    output = tmp_path / "bundle"
    monkeypatch.setattr(sys, "argv", ["arcade_run", "--reference", str(tmp_path / "reference.json"),
        "--decoder", str(tmp_path / "decoder.json"), "--trials", str(tmp_path / "half.json"),
        str(tmp_path / "quarter.json"), "--select", "intact", "half", "--output", str(output),
        "--rationale", "Selected examples, not a dose-response claim.", "--video-duration", "3.1"])
    arcade_run.main()
    provenance = json.loads((output / "provenance.json").read_text())
    assert len(provenance["conditions"]) == 3
    assert provenance["config"]["render"]["playback_speed"] == pytest.approx(2)
    assert json.loads((output / "render.json").read_text())["frames"] == 93
    assert provenance["artifacts"]["video"]["sha256"] == sha256_file(output / "demo.mp4")
    assert (output / "sources/00-reference.json").read_bytes() == (tmp_path / "reference.json").read_bytes()
    assert len(json.loads((output / "experiment.json").read_text())["episodes"]) == 2
    assert len(json.loads((output / "exploration.json").read_text())["episodes"]) == 3
    with pytest.raises(SystemExit):
        arcade_run.main()


@pytest.mark.parametrize("duration", ["3", "0", "nan", "inf", "3.11"])
def test_invalid_video_duration_fails_before_loading_data(duration, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["arcade_run", "--reference", "missing-reference.json",
        "--decoder", "missing-decoder.json", "--trials", "missing-trial.json", "--select", "intact", "half",
        "--output", str(tmp_path / "out"), "--rationale", "Test", "--video-duration", duration])
    with pytest.raises(SystemExit) as error:
        arcade_run.main()
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()
