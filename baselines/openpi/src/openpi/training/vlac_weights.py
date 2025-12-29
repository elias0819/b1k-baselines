import json
import logging
import re
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np
# Keep import explicit here to avoid NameError if this module is loaded in partial environments.
from openpi.shared import download as _download

LOGGER = logging.getLogger(__name__)

class VlacWeightTable(NamedTuple):
    weights: np.ndarray
    lengths: np.ndarray
    default_weight: float
    episode_index: np.ndarray

def _extract_episode_and_weight(entry: dict) -> tuple[int | None, object | None]:
    """Try to read an episode index and weight from a JSON entry."""
    episode = None
    weight = None

    for key in ("episode_index", "episode", "episode_id"):
        if key in entry:
            episode = entry[key]
            break

    for key in ("weight", "vlac_weight", "value"):
        if key in entry:
            weight = entry[key]
            break

    try:
        episode = None if episode is None else int(episode)
    except (TypeError, ValueError):
        episode = None

    return episode, weight

def _parse_episode_key(key: str | int) -> int | None:
    # 支持 "episode_00000010" 这类字符串，取末尾数字
    if isinstance(key, int):
        return key
    key = str(key).strip()
    match = re.search(r"(\d+)$", key)
    return int(match.group(1)) if match else None


def _array_from_value(value) -> np.ndarray | None:
    """Convert a JSON value into a 1D float32 array if possible."""
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=np.float32).flatten()
        return arr if arr.size > 0 else None
    try:
        return np.asarray([float(value)], dtype=np.float32)
    except (TypeError, ValueError):
        return None


def load_weight_mapping(weight_path: str) -> dict[int, np.ndarray]:
    local_path = Path(_download.maybe_download(weight_path))
    LOGGER.info("Loading VLAC weights from %s", local_path)

    if local_path.suffix.lower() == ".jsonl":
        entries: Iterable[dict] = (json.loads(line) for line in local_path.read_text().splitlines())
    else:
        entries = json.load(local_path.open("r"))

    mapping: dict[int, np.ndarray] = {}
    if isinstance(entries, dict):
        for raw_key, raw_value in entries.items():
            episode = _parse_episode_key(raw_key)
            weight = _array_from_value(raw_value)
            if episode is None or weight is None:
                continue
            mapping[episode] = weight
    elif isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            episode, weight = _extract_episode_and_weight(entry)
            if episode is None and len(entry) == 1:
                raw_key, raw_val = next(iter(entry.items()))
                episode = _parse_episode_key(raw_key)
                weight = _array_from_value(raw_val)
            else:
                weight = _array_from_value(weight)
            if episode is None or weight is None:
                continue
            mapping[episode] = weight
    else:
        raise ValueError(f"Unsupported VLAC weight file format: {type(entries)}")

    LOGGER.info("Loaded %d VLAC weights", len(mapping))
    return mapping


def build_weight_table(
    mapping: dict[int, np.ndarray],
    *,
    default_weight: float = 1.0,
    min_size: int = 0,
    chunk_size: int | None = None,
) -> VlacWeightTable | None:
    """Convert a mapping to a padded table of per-episode weight sequences."""
    if not mapping and min_size == 0:
        LOGGER.warning("No VLAC weights found and no minimum size requested; weighting will be skipped.")
        return None

    episode_ids = np.asarray(sorted(ep for ep in mapping.keys() if ep >= 0), dtype=np.int32)
    if episode_ids.size == 0:
        if min_size == 0:
            LOGGER.warning("No VLAC weights found and no minimum size requested; weighting will be skipped.")
            return None
        episode_ids = np.arange(min_size, dtype=np.int32)

    # Respect min_size by padding additional placeholder rows with default weights if requested.
    if min_size > episode_ids.size:
        extra_rows = np.arange(episode_ids.max() + 1, episode_ids.max() + 1 + (min_size - episode_ids.size), dtype=np.int32)
        episode_ids = np.concatenate([episode_ids, extra_rows], axis=0)

    sequences: list[np.ndarray] = []
    max_len = max(1, chunk_size or 0)
    for ep in episode_ids:
        seq = mapping.get(int(ep))
        if seq is None:
            sequences.append(np.asarray([default_weight], dtype=np.float32))
            continue
        seq_arr = np.asarray(seq, dtype=np.float32).flatten()
        if seq_arr.size == 0:
            sequences.append(np.asarray([default_weight], dtype=np.float32))
            continue
        sequences.append(seq_arr)
        max_len = max(max_len, seq_arr.shape[0])
    weights = np.full((len(sequences), max_len), default_weight, dtype=np.float32)
    lengths = np.full((len(sequences),), max_len if not mapping else 1, dtype=np.int32)
    for idx, seq in enumerate(sequences):
        seq_len = min(max_len, seq.shape[0])
        weights[idx, :seq_len] = seq[:seq_len]
        lengths[idx] = seq_len

    return VlacWeightTable(
        weights=weights,
        lengths=lengths,
        default_weight=float(default_weight),
        episode_index=episode_ids,
    )