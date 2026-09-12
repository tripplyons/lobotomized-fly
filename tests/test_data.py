"""Tiny schema-faithful fixtures; these tests never fetch a real dataset."""
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest

from flybird.data import FILES, download_data, load_graph
from flybird.provenance import build_provenance


@pytest.fixture
def fixture_data(tmp_path):
    feather.write_feather(pa.table({
        "bodyId": [30, 10, 20, 40], "status": ["Traced", "Traced", "Glia", "Traced"],
        "superclass": ["descending_neuron", "ol_sensory", None, "cb_intrinsic"],
        "class": [None, "visual", None, "CX"],
    }), tmp_path / FILES["annotations"])
    feather.write_feather(pa.table({
        "body_pre": [10, 10, 30, 20, 999, 10, 30],
        "body_post": [30, 30, 10, 10, 30, 10, 40],
        "weight": [2, 3, 4, 100, 50, 1, 0],
    }), tmp_path / FILES["connections"], chunksize=2)
    feather.write_feather(pa.table({"body": [10, 20, 30, 40],
                                   "consensus_nt": ["acetylcholine", None, "gaba", "glutamate"]}),
                          tmp_path / FILES["neurotransmitters"])
    return tmp_path


def test_import_filters_orientation_counts_and_signs(fixture_data):
    graph = load_graph(fixture_data)
    assert graph.node_ids.tolist() == [10, 30, 40]
    np.testing.assert_array_equal(graph.contacts.toarray(), [[1, 4, 0], [5, 0, 0], [0, 0, 0]])
    np.testing.assert_allclose(graph.weights.toarray(), [[.2, -.8, 0], [1, 0, 0], [0, 0, 0]])
    assert graph.weights.dtype == np.float32
    counts = graph.manifest["anatomical_counts"]
    assert counts == {"retained_nodes": 3, "retained_pair_edges": 3, "summed_contacts": 10,
                      "retained_source_rows": 4, "isolated_nodes": 1}
    assert graph.manifest["source_summed_contacts"] == 160
    assert graph.manifest["excluded_counts"]["summed_contacts"] == 150
    assert graph.manifest["row_counts"]["connections"] == 7
    assert graph.regions["descending"].tolist() == [False, True, False]
    assert graph.regions["sensory"].tolist() == [True, False, False]
    assert graph.regions["central_complex"].tolist() == [False, False, True]


def test_real_provenance_uses_actual_sources(fixture_data):
    from dataclasses import asdict
    import hashlib
    from flybird.brain import Condition, ConnectomeBrain
    from flybird.provenance import audit_conditions
    graph = load_graph(fixture_data)
    condition = Condition("intact", "Intact")
    config = {"seeds": [7], "conditions": [asdict(condition)]}
    run = {**config, "episodes": [{"condition": asdict(condition), "seed": 7,
                                   "brain": ConnectomeBrain(graph, condition, 7).manifest(),
                                   "score": 0, "course_sha256": hashlib.sha256(b"fixture").hexdigest()}]}
    result = build_provenance(seed=7, config=config, conditions=audit_conditions(graph, run),
                              synthetic_counts={"nodes": 0, "pair_edges": 0},
                              source_paths=[], dataset_manifest=graph.manifest)
    assert result["dataset"]["anatomical_counts"]["retained_nodes"] == 3
    assert len(result["dataset"]["imported_sources"]) == 3
    assert result["model_kind"] == "connectome_constrained_engineered_controller"
    assert not any("No MaleCNS" in text for text in result["limitations"])


def test_existing_downloads_do_not_contact_network(fixture_data, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("network forbidden")
    monkeypatch.setattr("urllib.request.urlopen", unexpected)
    assert all(path.exists() for path in download_data(fixture_data).values())
