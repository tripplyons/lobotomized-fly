import hashlib
import json

import pytest

from flybird.provenance import build_provenance, sha256_file, write_provenance


def ledger(**kwargs):
    options = dict(seed=7, config={"fps": 30}, synthetic_counts={"nodes": 4, "pair_edges": 3}, source_paths=[])
    options.update(kwargs)
    return build_provenance(**options)


def test_synthetic_counts_never_become_anatomy():
    result = ledger()
    assert result["dataset"]["anatomical_counts"] == {
        "retained_nodes": 0, "retained_pair_edges": 0, "summed_contacts": 0}
    assert result["dataset"]["imported_sources"] == []
    assert result["synthetic_counts"]["nodes"] == 4
    assert "not a fly brain" in result["display_label"]


def test_hashes_stable_sensitive_and_snapshot(tmp_path):
    source = tmp_path / "source.py"
    source.write_bytes(b"hello")
    config = {"b": 2, "a": 1}
    first = ledger(config=config, source_paths=[source], artifacts={"code": source})
    second = ledger(config={"a": 1, "b": 2}, source_paths=[source], artifacts={"code": source})
    assert first == second
    config["a"] = 9
    assert first["config"]["a"] == 1
    assert first["artifacts"]["code"]["sha256"] == hashlib.sha256(b"hello").hexdigest()
    source.write_bytes(b"changed")
    assert sha256_file(source) != first["artifacts"]["code"]["sha256"]
    assert ledger(config=config)["config_sha256"] != first["config_sha256"]


@pytest.mark.parametrize("counts", [{}, {"nodes": -1, "pair_edges": 0},
                                    {"nodes": True, "pair_edges": 0},
                                    {"nodes": 1, "pair_edges": 2}])
def test_invalid_counts_rejected(counts):
    with pytest.raises(ValueError):
        ledger(synthetic_counts=counts)


def test_bad_config_and_missing_source_fail(tmp_path):
    with pytest.raises(ValueError):
        ledger(config={"bad": float("nan")})
    with pytest.raises(FileNotFoundError):
        ledger(source_paths=[tmp_path / "missing"])


def test_incomplete_real_manifest_rejected():
    with pytest.raises(ValueError):
        ledger(dataset_manifest={"name": "MaleCNS"})


def test_write_keeps_negative_conditions(tmp_path):
    path = tmp_path / "run" / "provenance.json"
    result = write_provenance(path, seed=2, config={}, synthetic_counts={"nodes": 0, "pair_edges": 0},
                              source_paths=[], conditions=[{"name": "failed-null", "score": 0}])
    assert json.loads(path.read_text()) == result
    assert result["conditions"][0]["score"] == 0
