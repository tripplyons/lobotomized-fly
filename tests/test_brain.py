"""Small synthetic graph fixtures test mechanics, never biological claims."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from flybird.brain import (Condition, ConnectomeBrain, EngineeredBrain,
                           ReservoirConfig, TOTAL_UNITS, UNITS_PER_REGION,
                           connectome_conditions, fit_decoder)


def tiny_graph():
    return SimpleNamespace(
        node_ids=np.array([10, 20, 30, 40]),
        weights=sparse.csr_matrix(np.array([[0, 0, 0, 0], [1, 0, 0, 0],
                                          [0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.float32)),
        regions={"visual": np.array([True, False, False, False]),
                 "motor": np.array([False, False, True, False])},
        manifest={"fixture": True})


def test_exact_lesion_counts_and_mask_reproducibility():
    lesion = EngineeredBrain(Condition("lesion", "Lesion", "region", .75, "visual"), 7)
    null_condition = Condition("null", "Null", "matched_random", .75, "visual")
    null = EngineeredBrain(null_condition, 7)
    assert lesion.active.count(False) == null.active.count(False) == UNITS_PER_REGION
    assert null.active == EngineeredBrain(null_condition, 7).active
    assert null.active != EngineeredBrain(null_condition, 8).active
    zero = EngineeredBrain(Condition("zero", "Zero", "amount", 0))
    assert sum(zero.active) == 0
    assert not zero.act(.9, .8, .2)


def test_synthetic_rewire_preserves_degrees_and_negative_result():
    intact = EngineeredBrain(Condition("intact", "Intact"))
    rewired = EngineeredBrain(Condition("rewired", "Rewired", "rewired"))
    def degrees(brain):
        edges = brain.manifest()["edges"]
        return ([sum(a == i for a, b in edges) for i in range(TOTAL_UNITS)],
                [sum(b == i for a, b in edges) for i in range(TOTAL_UNITS)])
    assert degrees(intact) == degrees(rewired)
    assert intact.manifest()["edges"] != rewired.manifest()["edges"]
    for y in (.2, .5, .8):
        assert intact.act(y, .2, .5) == rewired.act(y, .2, .5)


def test_real_graph_direction_all_nodes_and_no_recurrence():
    graph = tiny_graph()
    cfg = ReservoirConfig(neural_steps=3, readout_groups=2)
    intact = ConnectomeBrain(graph, Condition("intact", "Intact"), config=cfg)
    no_edges = ConnectomeBrain(graph, Condition("null", "Null", "no_recurrence"), config=cfg)
    intact.observe(.8, .2, .4)
    no_edges.observe(.8, .2, .4)
    assert len(intact.state) == len(graph.node_ids)
    assert intact.state[2] != 0  # sensory -> interneuron -> motor
    assert no_edges.state[2] == 0
    assert intact.state[3] == 0  # retained isolate is still represented
    assert intact.manifest()["retained_pair_edges"] == 2
    assert no_edges.manifest()["retained_pair_edges"] == 0


def test_real_lesion_silences_input_and_output_without_weight_changes():
    graph = tiny_graph()
    original = graph.weights.copy()
    brain = ConnectomeBrain(graph, Condition("lesion", "Lesion", "region", 1, "visual"))
    for _ in range(4):
        brain.observe(.9, .3, .2)
    assert np.all(brain.state == 0)
    assert (graph.weights != original).nnz == 0
    assert brain.manifest()["removed_ids"] == [10]
    random = ConnectomeBrain(graph, Condition("null", "Null", "matched_random", 1, "visual"))
    assert random.active.sum() == brain.active.sum()


def test_decoder_needs_training_and_anatomical_masks_do_not_fallback():
    graph = tiny_graph()
    with pytest.raises(ValueError, match="no automatic random fallback"):
        ConnectomeBrain(graph, Condition("intact", "Intact"), input_regions=("unknown",))
    with pytest.raises(RuntimeError, match="trained"):
        ConnectomeBrain(graph, Condition("intact", "Intact")).act(.5, .2, .6)
    x = np.arange(100, dtype=float).reshape(50, 2) / 100
    decoder = fit_decoder(x, x[:, 0] * 2 - 1)
    assert decoder["training_rmse"] < .001
    assert decoder["samples"] == 50
    assert any(c.kind == "no_recurrence" for c in connectome_conditions(["visual"]))
