"""Official MaleCNS v1.0 download and bounded-memory sparse graph import."""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Any
import urllib.request

import numpy as np
import pyarrow.feather as feather
import pyarrow.ipc as ipc
from scipy import sparse

from .provenance import sha256_file

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "connections": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}


@dataclass
class Connectome:
    node_ids: np.ndarray
    weights: sparse.csr_matrix
    contacts: sparse.csr_matrix
    regions: dict[str, np.ndarray]
    manifest: dict[str, Any]


def download_data(directory: str | Path) -> dict[str, Path]:
    """Fetch missing official files atomically; never fetch synaptic point data."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for role, name in FILES.items():
        path = directory / name
        if not path.exists():
            temporary = path.with_suffix(".partial")
            try:
                with urllib.request.urlopen(BASE_URL + name, timeout=120) as response:
                    expected = response.headers.get("Content-Length")
                    with temporary.open("wb") as output:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                if expected is not None and temporary.stat().st_size != int(expected):
                    raise ValueError(f"Incomplete download: {name}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        result[role] = path
    return result


def annotation_regions(superclasses: np.ndarray, classes: np.ndarray) -> dict[str, np.ndarray]:
    """Reviewable population predicates, not inferred cognitive functions."""
    return {
        "descending": superclasses == "descending_neuron",
        "sensory": np.isin(superclasses, ["ol_sensory", "cb_sensory", "vnc_sensory"]),
        "visual": np.isin(superclasses, ["ol_intrinsic", "ol_sensory", "visual_projection", "visual_centrifugal"]),
        "central_complex": classes == "CX",
        "mushroom_body": np.isin(classes, ["Kenyon_Cell", "MBON", "DAN"]),
        "motor": np.isin(superclasses, ["descending_neuron", "vnc_motor", "cb_motor"]),
        "vnc": np.array([str(value).startswith("vnc_") for value in superclasses]),
    }


def load_graph(directory: str | Path, *, download: bool = False) -> Connectome:
    """Retain every status=Traced neuron and all positive contacts between them.

    Rows are postsynaptic, columns presynaptic. ``contacts`` retains integer
    anatomical contact counts; ``weights`` divides each incoming row by its
    contact total and applies presynaptic signs (GABA/glutamate negative,
    everything else positive). This sign/normalization is a model assumption.
    The source Feather is streamed in record batches, not materialized whole.
    """
    directory = Path(directory)
    paths = download_data(directory) if download else {k: directory / v for k, v in FILES.items()}
    annotations = feather.read_table(paths["annotations"])
    selected = np.asarray(annotations["status"].to_pylist(), dtype=object) == "Traced"
    ids = annotations["bodyId"].to_numpy()[selected]
    order = np.argsort(ids)
    ids = ids[order]
    if len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError("Expected nonempty unique Traced body IDs")
    superclasses = np.asarray(annotations["superclass"].to_pylist(), dtype=object)[selected][order]
    classes = np.asarray(annotations["class"].to_pylist(), dtype=object)[selected][order]
    regions = annotation_regions(superclasses, classes)
    reader = ipc.open_file(paths["connections"])
    pre_parts, post_parts, weight_parts = [], [], []
    row_count = 0
    source_contacts = 0
    nonpositive_rows = 0
    for batch_index in range(reader.num_record_batches):
        batch = reader.get_batch(batch_index)
        pre = batch.column("body_pre").to_numpy()
        post = batch.column("body_post").to_numpy()
        count = batch.column("weight").to_numpy()
        row_count += len(pre)
        source_contacts += int(count.sum())
        nonpositive_rows += int((count <= 0).sum())
        a, b = np.searchsorted(ids, pre), np.searchsorted(ids, post)
        valid = (a < len(ids)) & (b < len(ids)) & (count > 0)
        a, b, count, pre, post = a[valid], b[valid], count[valid], pre[valid], post[valid]
        valid = (ids[a] == pre) & (ids[b] == post)
        pre_parts.append(a[valid].astype(np.int32))
        post_parts.append(b[valid].astype(np.int32))
        weight_parts.append(count[valid].astype(np.int64))
    raw_count = sum(len(part) for part in weight_parts)
    contacts = sparse.coo_matrix((np.concatenate(weight_parts),
                                 (np.concatenate(post_parts), np.concatenate(pre_parts))),
                                shape=(len(ids), len(ids)), dtype=np.int64).tocsr()
    del pre_parts, post_parts, weight_parts
    contacts.sum_duplicates()
    nt = feather.read_table(paths["neurotransmitters"], columns=["body", "consensus_nt"])
    nt_ids = nt["body"].to_numpy()
    nt_labels = np.asarray(nt["consensus_nt"].to_pylist(), dtype=object)
    retained_nt = np.isin(nt_ids, ids)
    label_counts = Counter(str(label) if label is not None else "<missing>" for label in nt_labels[retained_nt])
    missing_nt_bodies = int((~np.isin(ids, nt_ids)).sum())
    signs = np.ones(len(ids), dtype=np.float32)
    inhibitory_ids = nt_ids[np.isin(nt_labels, ["gaba", "GABA", "glutamate"])]
    signs[np.isin(ids, inhibitory_ids)] = -1
    weights = contacts.astype(np.float32)
    totals = np.asarray(contacts.sum(axis=1)).ravel()
    # Operate row chunks to avoid constructing another full sparse matrix.
    for start in range(0, len(ids), 4096):
        end = min(start + 4096, len(ids))
        lo, hi = weights.indptr[start], weights.indptr[end]
        weights.data[lo:hi] *= signs[weights.indices[lo:hi]]
        weights.data[lo:hi] /= np.repeat(np.maximum(totals[start:end], 1), np.diff(weights.indptr[start:end + 1]))
    row_counts = {"annotations": annotations.num_rows, "neurotransmitters": nt.num_rows,
                  "connections": row_count}
    manifest = {
        "name": "MaleCNS", "version": "v1.0", "license": "CC-BY-4.0",
        "attribution": "MaleCNS Consortium / Janelia Research Campus; https://male-cns.janelia.org/",
        "source_urls": [BASE_URL + name for name in FILES.values()],
        "imported_sources": [{"role": role, "filename": path.name, "url": BASE_URL + path.name,
                              "bytes": path.stat().st_size, "sha256": sha256_file(path),
                              "rows": row_counts[role]} for role, path in paths.items()],
        "row_counts": row_counts,
        "neurotransmitters": {
            "retained_label_counts": dict(sorted(label_counts.items())),
            "observed_label_to_sign": {label: (-1 if label in ("gaba", "GABA", "glutamate") else 1)
                                       for label in sorted(label_counts)},
            "missing_table_bodies": missing_nt_bodies,
            "missing_consensus_labels": label_counts.get("<missing>", 0),
            "unknown_consensus_labels": sum(count for label, count in label_counts.items()
                                             if label.lower() in ("unknown", "unclear", "other", "")),
            "fallback_sign": 1,
        },
        "filters": {"minimum_confidence": 0.5, "body_status": "Traced",
                    "included_body_classes": "all classes with status Traced, including isolates",
                    "pair_edge_aggregation": "sum contacts over identical directed body pairs",
                    "minimum_contact_count": 1, "autapse_handling": "retained",
                    "transmitter_sign_rule": "consensus_nt GABA/glutamate negative; all others positive (modeled)",
                    "weight_normalization": "divide by total incoming contacts per postsynaptic neuron"},
        "anatomical_counts": {"retained_nodes": len(ids), "retained_pair_edges": contacts.nnz,
                              "summed_contacts": int(contacts.sum()), "retained_source_rows": raw_count,
                              "isolated_nodes": int(((np.diff(contacts.indptr) == 0) &
                                                     (np.asarray(contacts.getnnz(axis=0)) == 0)).sum())},
        "excluded_counts": {"annotation_rows_not_traced": annotations.num_rows - len(ids),
                            "connection_rows": row_count - raw_count,
                            "summed_contacts": source_contacts - int(contacts.sum()),
                            "source_nonpositive_rows": nonpositive_rows},
        "source_summed_contacts": source_contacts,
        "region_counts": {name: int(mask.sum()) for name, mask in regions.items()},
        "region_predicates": {
            "descending": "superclass == descending_neuron",
            "sensory": "superclass in ol_sensory, cb_sensory, vnc_sensory",
            "visual": "superclass in ol_intrinsic, ol_sensory, visual_projection, visual_centrifugal",
            "central_complex": "class == CX",
            "mushroom_body": "class in Kenyon_Cell, MBON, DAN (annotation-defined proxy; DAN not exclusively MB)",
            "motor": "superclass in descending_neuron, vnc_motor, cb_motor (output proxy)",
            "vnc": "superclass starts with vnc_",
        },
    }
    return Connectome(ids, weights, contacts, regions, manifest)
