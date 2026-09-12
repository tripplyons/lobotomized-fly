"""An explicitly engineered rate controller, NOT a MaleCNS simulation.

Four synthetic modules encode gap offset, body position, falling velocity and
motor drive. Lesions silence units in the same computation used by the intact
controller. There are no condition-specific physics, errors or death timers.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random


REGIONS = ("visual", "position", "prediction", "motor")
UNITS_PER_REGION = 32
TOTAL_UNITS = len(REGIONS) * UNITS_PER_REGION
MODEL_VERSION = "engineered-rate-controller-v1"


@dataclass(frozen=True)
class Condition:
    id: str
    label: str
    kind: str = "intact"
    retained_fraction: float = 1.0
    lesion_region: str | None = None

    def __post_init__(self):
        if not self.id or not self.label:
            raise ValueError("condition id and label must be nonempty")
        if self.kind not in {"intact", "amount", "region", "matched_random", "rewired", "no_recurrence"}:
            raise ValueError("unknown condition kind")
        if not math.isfinite(self.retained_fraction) or not 0 <= self.retained_fraction <= 1:
            raise ValueError("retained_fraction must be between zero and one")
        if self.kind in {"region", "matched_random"} and not self.lesion_region:
            raise ValueError("region and matched_random conditions require a region")


def default_conditions():
    """Keep all controls, including an intentionally redundant wiring null."""
    conditions = [Condition("intact", "Intact engineered controller")]
    for percent in (75, 50, 25):
        conditions.append(Condition(f"amount_{percent}", f"{percent}% units retained", "amount", percent / 100))
    for region in REGIONS:
        conditions.append(Condition(f"without_{region}", f"{region.title()} module silenced", "region", .75, region))
        conditions.append(Condition(f"random_for_{region}", f"Random removal: {region} size", "matched_random", .75, region))
    conditions.append(Condition("rewired", "Degree-preserving wiring null", "rewired"))
    return conditions


def _seed(seed, namespace):
    return int.from_bytes(hashlib.sha256(f"{seed}:{namespace}".encode()).digest()[:8], "big")


class EngineeredBrain:
    """Fixed-population-normalized rate circuit with silent lesioned units.

    Engineered input channels: .5-gap_y, y-.5, .18*vy. Each is copied to
    32 rate units; matching-index units project to 32 motor units. Motor rates
    are sigmoid(20*(sum(inputs)-.10)); a fixed-population mean > .5 flaps.
    No homeostatic renormalization, training or biological parameters are used.
    The rewire independently permutes each module's matching-index projection,
    preserving every node's in/out degree. Redundant intact feature encoding
    makes it an expected negative result, not evidence for wiring specificity.
    """

    def __init__(self, condition: Condition, seed: int = 7):
        self.condition = condition
        self.active = [True] * TOTAL_UNITS
        rng = random.Random(_seed(seed, "lesion:" + condition.id))
        if condition.kind == "region":
            start = REGIONS.index(condition.lesion_region) * UNITS_PER_REGION
            removed = range(start, start + UNITS_PER_REGION)
        elif condition.kind in {"amount", "matched_random"}:
            count = UNITS_PER_REGION if condition.kind == "matched_random" else TOTAL_UNITS - round(TOTAL_UNITS * condition.retained_fraction)
            removed = rng.sample(range(TOTAL_UNITS), count)
        else:
            removed = []
        for index in removed:
            self.active[index] = False
        self.projections = [list(range(UNITS_PER_REGION)) for _ in range(3)]
        if condition.kind == "rewired":
            wire_rng = random.Random(_seed(seed, "rewire"))
            for projection in self.projections:
                wire_rng.shuffle(projection)
        self.last_activity = {region: 0.0 for region in REGIONS}

    def act(self, y: float, vy: float, gap_y: float):
        features = (.5 - gap_y, y - .5, .18 * vy)
        rates = []
        for module, value in enumerate(features):
            rates.append([value if self.active[module * UNITS_PER_REGION + i] else 0.0 for i in range(UNITS_PER_REGION)])
        motor = []
        for i in range(UNITS_PER_REGION):
            drive = sum(rates[module][self.projections[module][i]] for module in range(3)) - .10
            rate = 1 / (1 + math.exp(-max(-60, min(60, 20 * drive))))
            motor.append(rate if self.active[3 * UNITS_PER_REGION + i] else 0.0)
        for module, region in enumerate(REGIONS[:3]):
            self.last_activity[region] = sum(min(1, abs(value) * 4) for value in rates[module]) / UNITS_PER_REGION
        self.last_activity["motor"] = sum(motor) / UNITS_PER_REGION
        return self.last_activity["motor"] > .5

    def manifest(self):
        nodes = [{"id": i, "region": REGIONS[i // UNITS_PER_REGION], "active": active} for i, active in enumerate(self.active)]
        edges = [[module * UNITS_PER_REGION + self.projections[module][i], 3 * UNITS_PER_REGION + i] for module in range(3) for i in range(UNITS_PER_REGION)]
        retained_edges = [edge for edge in edges if all(self.active[i] for i in edge)]
        encoded = json.dumps({"nodes": nodes, "edges": edges}, sort_keys=True, separators=(",", ":")).encode()
        return {
            "model": MODEL_VERSION, "source": "synthetic engineered circuit; no connectome data imported",
            "condition": asdict(self.condition), "nodes": nodes, "edges": edges,
            "total_units": TOTAL_UNITS, "retained_units": sum(self.active),
            "retained_fraction": sum(self.active) / TOTAL_UNITS,
            "removed_ids": [i for i, active in enumerate(self.active) if not active],
            "region_retained": {region: sum(self.active[k * UNITS_PER_REGION:(k + 1) * UNITS_PER_REGION]) for k, region in enumerate(REGIONS)},
            "pair_edges": len(edges), "retained_pair_edges": len(retained_edges),
            "synaptic_contacts": None, "sha256": hashlib.sha256(encoded).hexdigest(),
            "null_scope": "Random controls match unit count only, not degree or anatomical region; rewired control preserves node degree.",
        }


@dataclass(frozen=True)
class ReservoirConfig:
    """Declared engineered dynamics; none are measured fly parameters."""
    recurrent_gain: float = .95
    input_gain: float = .75
    leak: float = 1.0
    neural_steps: int = 2
    readout_groups: int = 128
    encoder_seed: int = 2026
    ridge: float = .01


def connectome_conditions(regions):
    conditions = [Condition("intact", "Intact retained MaleCNS model")]
    conditions.extend(Condition(f"amount_{p}", f"{p}% neurons retained", "amount", p / 100) for p in (75, 50, 25))
    for region in regions:
        conditions.extend((Condition(f"without_{region}", f"{region.replace('_', ' ').title()} silenced", "region", 1, region),
                           Condition(f"random_for_{region}", f"Random: {region.replace('_', ' ')} size", "matched_random", 1, region)))
    conditions.append(Condition("no_recurrence", "No recurrent connections", "no_recurrence"))
    return conditions


class ConnectomeBrain:
    """All retained graph nodes updated; anatomy-restricted engineered I/O.

    ``graph.weights`` must be signed, absolute-row-normalized CSR POST,PRE.
    Lesions silence state/input/output without renormalizing remaining weights.
    Only visual/sensory neurons receive engineered features. The fitted decoder
    sees pooled descending/motor neural state, never raw game coordinates.
    """

    def __init__(self, graph, condition, seed=7, *, config=None, decoder=None,
                 input_regions=("visual",), output_regions=("motor",)):
        import numpy as np
        self.graph, self.condition = graph, condition
        self.config = config or ReservoirConfig()
        cfg = self.config
        if not 0 < cfg.leak <= 1 or cfg.neural_steps < 1 or cfg.readout_groups < 1:
            raise ValueError("invalid reservoir parameters")
        n = len(graph.node_ids)
        if graph.weights.shape != (n, n):
            raise ValueError("graph node count and matrix shape disagree")
        self.active = np.ones(n, dtype=bool)
        rng = np.random.default_rng(_seed(seed, "lesion:" + condition.id))
        if condition.kind in {"region", "matched_random"}:
            if condition.lesion_region not in graph.regions:
                raise ValueError(f"unknown anatomical region {condition.lesion_region}")
            removed = np.flatnonzero(graph.regions[condition.lesion_region])
            if condition.kind == "matched_random":
                removed = rng.choice(n, size=len(removed), replace=False)
            self.active[removed] = False
        elif condition.kind == "amount":
            self.active[rng.choice(n, size=n - round(n * condition.retained_fraction), replace=False)] = False
        elif condition.kind == "rewired":
            raise ValueError("real-graph rewiring is not implemented; use the explicit no_recurrence control")
        def union(names):
            mask = np.zeros(n, dtype=bool)
            for name in names:
                if name not in graph.regions:
                    raise ValueError(f"missing anatomical I/O region {name}; no automatic random fallback")
                mask |= graph.regions[name]
            if not mask.any():
                raise ValueError("anatomical I/O mask is empty")
            return mask
        self.input_regions, self.output_regions = tuple(input_regions), tuple(output_regions)
        self.input_indices = np.flatnonzero(union(input_regions))
        self.output_indices = np.flatnonzero(union(output_regions))
        encoder_rng = np.random.default_rng(cfg.encoder_seed)
        self.encoder = encoder_rng.choice(np.array([-1., 1.], dtype=np.float32), size=(len(self.input_indices), 3))
        self.output_groups = np.arange(len(self.output_indices)) % cfg.readout_groups
        self.group_counts = np.maximum(1, np.bincount(self.output_groups, minlength=cfg.readout_groups))
        self.state = np.zeros(n, dtype=np.float32)
        self.decoder = decoder
        self.last_features = np.zeros(cfg.readout_groups, dtype=np.float64)
        self.last_activity = {name: 0.0 for name in graph.regions}
        self.steps = 0

    def observe(self, y, vy, gap_y):
        import numpy as np
        cfg = self.config
        features = np.array([y - .5, .5 - gap_y, .18 * vy], dtype=np.float32)
        drive = cfg.input_gain * (self.encoder @ features)
        for _ in range(cfg.neural_steps):
            propagated = np.zeros_like(self.state) if self.condition.kind == "no_recurrence" else self.graph.weights @ self.state
            propagated *= cfg.recurrent_gain
            propagated[self.input_indices] += drive
            self.state *= 1 - cfg.leak
            self.state += cfg.leak * np.tanh(propagated)
            self.state[~self.active] = 0
            self.steps += 1
        self.last_features = np.bincount(self.output_groups, weights=self.state[self.output_indices], minlength=cfg.readout_groups) / self.group_counts
        self.last_activity = {name: float(np.mean(np.abs(self.state[mask]))) if mask.any() else 0.0 for name, mask in self.graph.regions.items()}
        return self.last_features

    def act(self, y, vy, gap_y):
        import numpy as np
        features = self.observe(y, vy, gap_y)
        if self.decoder is None:
            raise RuntimeError("connectome decoder must be trained before game evaluation")
        normalized = (features - np.asarray(self.decoder["mean"])) / np.asarray(self.decoder["scale"])
        return float(normalized @ np.asarray(self.decoder["weights"]) + self.decoder["bias"]) > 0

    def manifest(self):
        import numpy as np
        mask_bytes = np.packbits(self.active).tobytes()
        # Counts use CSR row masks without constructing another full sparse graph.
        retained_edges = 0
        contacts = getattr(self.graph, "contacts", None)
        retained_contacts = 0 if contacts is not None else None
        for start in range(0, len(self.active), 4096):
            end = min(len(self.active), start + 4096)
            lo, hi = self.graph.weights.indptr[start], self.graph.weights.indptr[end]
            row_active = np.repeat(self.active[start:end], np.diff(self.graph.weights.indptr[start:end + 1]))
            retained = row_active & self.active[self.graph.weights.indices[lo:hi]]
            retained_edges += int(np.count_nonzero(retained))
            if contacts is not None:
                retained_contacts += int(contacts.data[lo:hi][retained].sum())
        if self.condition.kind == "no_recurrence":
            retained_edges = 0
            retained_contacts = 0
        return {"model": "malecns-engineered-rate-reservoir-v1", "source": "official MaleCNS retained graph with engineered rate dynamics and task interfaces",
                "condition": asdict(self.condition), "total_units": len(self.active), "retained_units": int(self.active.sum()),
                "retained_fraction": float(self.active.mean()), "pair_edges": int(self.graph.weights.nnz),
                "retained_pair_edges": retained_edges, "retained_synaptic_contacts": retained_contacts,
                "sha256": hashlib.sha256(mask_bytes).hexdigest(),
                "mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
                "removed_ids": self.graph.node_ids[~self.active].tolist(),
                "region_retained": {name: int(np.count_nonzero(mask & self.active)) for name, mask in self.graph.regions.items()},
                "region_totals": {name: int(np.count_nonzero(mask)) for name, mask in self.graph.regions.items()},
                "input_regions": list(self.input_regions), "output_regions": list(self.output_regions),
                "input_count": len(self.input_indices), "output_count": len(self.output_indices),
                "parameters": asdict(self.config), "neural_steps_executed": self.steps,
                "dataset_manifest": self.graph.manifest,
                "decoder_sha256": hashlib.sha256(json.dumps(self.decoder, sort_keys=True).encode()).hexdigest() if self.decoder else None,
                "null_scope": "Random lesions match neuron count only; no_recurrence removes all recurrent connections. No biological behavior validation."}


def fit_decoder(feature_rows, targets, config=None):
    """Ridge fit to an explicitly engineered teacher; no direct input bypass."""
    import numpy as np
    cfg = config or ReservoirConfig()
    x = np.asarray(feature_rows, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or len(target) != len(x):
        raise ValueError("training requires aligned neural feature rows and targets")
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-8)
    normalized = (x - mean) / scale
    bias = float(target.mean())
    weights = np.linalg.solve(normalized.T @ normalized + cfg.ridge * np.eye(x.shape[1]), normalized.T @ (target - bias))
    predicted = normalized @ weights + bias
    return {"mean": mean.tolist(), "scale": scale.tolist(), "weights": weights.tolist(), "bias": bias,
            "training_rmse": float(np.sqrt(np.mean((predicted - target) ** 2))),
            "training_sign_accuracy": float(np.mean((predicted > 0) == (target > 0))),
            "samples": len(x), "method": "ridge imitation of engineered y+.18*vy-gap_y-.10 teacher"}
