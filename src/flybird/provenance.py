"""Deterministic provenance separating anatomical data from engineered models."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence


LIMITATIONS = (
    "No MaleCNS or FlyWire neurons, pair edges, or synaptic contacts are imported.",
    "100% or intact means the complete synthetic controller, not a full fly brain.",
    "Region names denote engineered functional modules, not anatomical cell sets.",
    "Game-state encoding, network parameters, action decoding, and game physics are engineered.",
    "Successful play demonstrates an engineered controller, not biological intelligence or learning.",
    "Lesion and null results concern this model only; retain failed and negative results.",
    "One demonstration seed cannot establish a robust or anatomical causal effect.",
)

REAL_LIMITATIONS = (
    "Full means all official status=Traced MaleCNS v1.0 neurons, not every segmented body or a living brain.",
    "Contact counts are reconstructed anatomy, not measured synaptic conductances.",
    "Neurotransmitter signs, rate dynamics, normalization, game encoder and action decoder are modeled or engineered.",
    "Region masks are annotation predicates and can overlap; motor and mushroom-body groups are proxies.",
    "Game success can come from task-specific training and does not establish fly cognition or biological validation.",
    "Keep all evaluated lesions, null controls, failed trials and train/evaluation seeds in the run record.",
    "A single seed or comparison cannot establish a robust anatomical causal effect.",
)


def canonical_json(value: Any) -> bytes:
    """Stable, strict JSON bytes; reject NaN instead of emitting invalid JSON."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return {"filename": path.name, "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def build_provenance(
    *,
    seed: int,
    config: Mapping[str, Any],
    synthetic_counts: Mapping[str, int],
    artifacts: Mapping[str, str | Path] | None = None,
    source_paths: Sequence[str | Path] | None = None,
    conditions: Sequence[Mapping[str, Any]] = (),
    dataset_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an auditable run ledger without timestamps or absolute paths.

    ``synthetic_counts`` must include ``nodes`` and ``pair_edges`` for the
    intact model. Additional integer counters are allowed, but anatomical
    contacts are never inferred from synthetic weights. ``conditions`` holds
    all evaluated variants, including unsuccessful controls. ``source_paths``
    are implementation files, never anatomical data files. Artifacts must
    already exist; omit provenance.json itself to avoid a circular hash.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    counts = dict(synthetic_counts)
    if not {"nodes", "pair_edges"}.issubset(counts):
        raise ValueError("synthetic_counts requires nodes and pair_edges")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in counts.values()):
        raise ValueError("synthetic counts must be nonnegative integers")
    if counts["pair_edges"] > counts["nodes"] ** 2:
        raise ValueError("directed pair edges cannot exceed nodes squared")
    # Snapshot caller-owned structures and validate strict JSON serializability.
    config_copy = json.loads(canonical_json(dict(config)))
    condition_copy = json.loads(canonical_json(list(conditions)))
    paths = sorted(Path(__file__).parent.glob("*.py")) if source_paths is None else source_paths
    sources = [_file_record(p) for p in paths]
    sources.sort(key=lambda item: (item["filename"], item["sha256"]))
    record = {
        "schema_version": 1,
        "model_kind": "synthetic_engineered_controller",
        "display_label": "Intact synthetic model (not a fly brain)",
        "seed": seed,
        "config": config_copy,
        "config_sha256": hashlib.sha256(canonical_json(config_copy)).hexdigest(),
        "implementation_sources": sources,
        "implementation_sha256": hashlib.sha256(canonical_json(sources)).hexdigest(),
        "runtime": {"python": platform.python_version()},
        "dataset": {
            "name": "synthetic", "version": "flybird-synthetic-v1",
            "imported_sources": [], "source_urls": [], "row_counts": {},
            "filters": {"minimum_confidence": None, "included_body_classes": [],
                        "pair_edge_aggregation": "not applicable: synthetic model",
                        "minimum_contact_count": None, "autapse_handling": "not anatomical",
                        "transmitter_sign_rule": "engineered, not transmitter predictions"},
            "anatomical_counts": {"retained_nodes": 0, "retained_pair_edges": 0,
                                  "summed_contacts": 0},
        },
        "synthetic_counts": counts,
        "conditions": condition_copy,
        "artifacts": {name: _file_record(path) for name, path in sorted((artifacts or {}).items())},
        "claim_ledger": [
            {"part": "nodes and connectivity", "classification": "engineered",
             "confidence": "verified", "claim": "Generated model, not reconstructed neurons or synapses."},
            {"part": "functional regions", "classification": "engineered",
             "confidence": "verified", "claim": "Module labels are not MaleCNS anatomical annotations."},
            {"part": "input encoder and action decoder", "classification": "engineered",
             "confidence": "verified", "claim": "Game variables and flap decisions use task-specific adapters."},
            {"part": "dynamics and controller parameters", "classification": "modeled and engineered",
             "confidence": "verified", "claim": "Not measured membrane constants or synaptic conductances."},
            {"part": "game behavior", "classification": "engineered",
             "confidence": "repo claim", "claim": "Scores apply only to the recorded game, configuration, and seeds."},
            {"part": "biological validation", "classification": "unsupported",
             "confidence": "not found", "claim": "No biological validation is provided by this demo."},
        ],
        "limitations": list(LIMITATIONS),
    }
    if dataset_manifest is not None:
        dataset = json.loads(canonical_json(dict(dataset_manifest)))
        _validate_dataset(dataset)
        _validate_conditions(condition_copy, config_copy, dataset)
        record["dataset"] = dataset
        record["model_kind"] = "connectome_constrained_engineered_controller"
        record["display_label"] = "Full traced MaleCNS model (engineered dynamics and readout)"
        record["limitations"] = list(REAL_LIMITATIONS)
        record["claim_ledger"][0] = {
            "part": "nodes and connectivity", "classification": "reconstructed anatomy",
            "confidence": "verified", "claim": "Official MaleCNS v1.0 sources; retained counts and filters are in dataset manifest."}
        record["claim_ledger"][1] = {
            "part": "region masks", "classification": "curated or inferred annotations plus engineered grouping",
            "confidence": "verified", "claim": "Exact official annotation predicates recorded; not validated game-function mappings."}
        record["claim_ledger"].append({
            "part": "neurotransmitter sign", "classification": "predicted plus modeled",
            "confidence": "verified", "claim": "Official consensus_nt predictions mapped to modeled presynaptic signs."})
    return record


def _validate_dataset(dataset: Mapping[str, Any]) -> None:
    required = {"name", "version", "license", "attribution", "imported_sources",
                "source_urls", "row_counts", "filters", "anatomical_counts"}
    if not required.issubset(dataset) or not dataset["imported_sources"]:
        raise ValueError("Real dataset manifest requires sources, hashes, counts, filters and attribution")
    for source in dataset["imported_sources"]:
        digest = source.get("sha256", "")
        if (not source.get("url") or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or not isinstance(source.get("rows"), int) or source["rows"] < 0
                or not isinstance(source.get("bytes"), int) or source["bytes"] <= 0):
            raise ValueError("Every imported source needs URL, SHA-256, bytes and row count")
    counts = dataset["anatomical_counts"]
    for name in ("retained_nodes", "retained_pair_edges", "summed_contacts"):
        if isinstance(counts.get(name), bool) or not isinstance(counts.get(name), int) or counts[name] < 0:
            raise ValueError("Anatomical counts must be explicit nonnegative integers")


def audit_conditions(graph, experiment: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Audit every actual episode mask against raw POST,PRE int64 contacts.

    Anatomy remaining after node removal is distinct from effective recurrence:
    the no_recurrence null retains anatomy but disables all recurrent pairs.
    Summations use Python integers to avoid int64 overflow; no float weights.
    """
    import numpy as np

    ids = graph.node_ids.tolist()
    index = {body: i for i, body in enumerate(ids)}
    contacts = graph.contacts
    if (len(index) != len(ids) or contacts.shape != (len(ids), len(ids))
            or contacts.dtype != np.dtype("int64") or not contacts.has_canonical_format
            or np.any(contacts.data <= 0)):
        raise ValueError("Expected unique nodes and canonical positive int64 raw contacts")
    totals = {"nodes": len(ids), "pair_edges": int(contacts.nnz),
              "contacts": sum(map(int, contacts.data))}
    expected = graph.manifest["anatomical_counts"]
    if list(totals.values()) != [expected[k] for k in ("retained_nodes", "retained_pair_edges", "summed_contacts")]:
        raise ValueError("Raw graph totals disagree with dataset manifest")
    definitions = {c["id"]: c for c in experiment["conditions"]}
    ledger = []
    for episode in experiment["episodes"]:
        condition, brain = episode["condition"], episode["brain"]
        if definitions.get(condition["id"]) != condition or brain["condition"] != condition:
            raise ValueError("Episode condition disagrees with declared condition or brain")
        removed = brain["removed_ids"]
        if any(type(body) is not int for body in removed) or len(set(removed)) != len(removed) or any(body not in index for body in removed):
            raise ValueError("Removed IDs must be unique graph body IDs")
        active = np.ones(len(ids), dtype=bool)
        active[[index[body] for body in removed]] = False
        kind = condition["kind"]
        if kind in {"region", "matched_random"}:
            region = graph.regions[condition["lesion_region"]]
            if (len(removed) != int(region.sum())
                    or (kind == "region" and not np.array_equal(~active, region))):
                raise ValueError("Actual mask does not match anatomical condition")
        elif kind == "amount":
            if int(active.sum()) != round(len(ids) * condition["retained_fraction"]):
                raise ValueError("Actual mask does not match retained amount")
        elif kind not in {"intact", "no_recurrence"} or removed:
            raise ValueError("Unsupported real condition or unexpected intact/null lesion")
        mask_hash = hashlib.sha256(np.packbits(active).tobytes()).hexdigest()
        if brain["mask_sha256"] != mask_hash or brain["retained_units"] != int(active.sum()):
            raise ValueError("Brain mask identity/count disagrees with removed IDs")
        pairs = raw_contacts = 0
        for start in range(0, len(ids), 4096):
            end = min(start + 4096, len(ids))
            lo, hi = contacts.indptr[start], contacts.indptr[end]
            keep = np.repeat(active[start:end], np.diff(contacts.indptr[start:end + 1])) & active[contacts.indices[lo:hi]]
            pairs += int(np.count_nonzero(keep))
            raw_contacts += sum(map(int, contacts.data[lo:hi][keep]))
        retained = {"nodes": int(active.sum()), "pair_edges": pairs, "contacts": raw_contacts}
        disabled = condition["kind"] == "no_recurrence"
        ledger.append({"condition": condition, "seed": episode["seed"],
                       "removed_ids": sorted(removed), "mask_sha256": mask_hash,
                       "removed_ids_sha256": hashlib.sha256(canonical_json(sorted(removed))).hexdigest(),
                       "node_order_sha256": hashlib.sha256(canonical_json(ids)).hexdigest(),
                       "mask_encoding": "numpy.packbits(active in graph node order), big bit order",
                       "total": totals, "retained": retained,
                       "removed": {k: totals[k] - retained[k] for k in totals},
                       "effective_recurrent_pair_edges": 0 if disabled else pairs,
                       "effective_recurrent_contacts": 0 if disabled else raw_contacts,
                       "recurrence_interpretation": "disabled; anatomy retained" if disabled else "retained anatomy used with engineered signed normalized weights",
                       "score": episode["score"], "course_sha256": episode["course_sha256"]})
    _validate_conditions(ledger, {"seeds": experiment["seeds"], "conditions": experiment["conditions"]}, graph.manifest)
    return ledger


def _validate_conditions(conditions, config, dataset):
    seeds, definitions = config.get("seeds", []), config.get("conditions", [])
    if (not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds)
            or not definitions or len({c["id"] for c in definitions}) != len(definitions)):
        raise ValueError("Real runs require unique seeds and condition definitions")
    declared = {c["id"]: c for c in definitions}
    expected = {(s, c) for s in seeds for c in declared}
    seen = set()
    counts = dataset["anatomical_counts"]
    totals = dict(zip(("nodes", "pair_edges", "contacts"),
                      (counts["retained_nodes"], counts["retained_pair_edges"], counts["summed_contacts"])))
    try:
        for item in conditions:
            condition = item["condition"]
            key = (item["seed"], condition["id"])
            if type(item["seed"]) is not int or key not in expected or key in seen or condition != declared[condition["id"]]:
                raise ValueError("Unexpected or duplicate episode condition")
            seen.add(key)
            if item["total"] != totals:
                raise ValueError("Condition graph totals disagree with dataset")
            for name in totals:
                values = [item[group][name] for group in ("total", "retained", "removed")]
                if any(type(v) is not int or v < 0 for v in values) or values[1] + values[2] != values[0]:
                    raise ValueError("Condition counts must conserve exact integer totals")
            removed = item["removed_ids"]
            if (any(type(v) is not int for v in removed) or removed != sorted(set(removed))
                    or len(removed) != item["removed"]["nodes"]
                    or item["removed_ids_sha256"] != hashlib.sha256(canonical_json(removed)).hexdigest()):
                raise ValueError("Invalid removed body ID ledger")
            for name in ("mask_sha256", "node_order_sha256", "course_sha256"):
                digest = item[name]
                if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("Missing or invalid identity hash")
            disabled = condition["kind"] == "no_recurrence"
            if condition["kind"] not in {"intact", "amount", "region", "matched_random", "no_recurrence"}:
                raise ValueError("Unsupported real condition kind")
            if condition["kind"] in {"intact", "no_recurrence"} and item["retained"] != totals:
                raise ValueError("Intact and no_recurrence conditions retain all anatomy")
            for name in ("pair_edges", "contacts"):
                value = item["effective_recurrent_" + name]
                if type(value) is not int or value != (0 if disabled else item["retained"][name]):
                    raise ValueError("Effective recurrence does not match null interpretation")
            if not item["recurrence_interpretation"] or type(item["score"]) is not int or item["score"] < 0:
                raise ValueError("Missing interpretation or invalid score")
    except (KeyError, TypeError) as error:
        raise ValueError("Real provenance requires a complete audited episode ledger") from error
    if seen != expected:
        raise ValueError("Missing episode ledger entries")


def write_provenance(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Write and return a provenance ledger, creating its output directory."""
    record = build_provenance(**kwargs)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    return record
