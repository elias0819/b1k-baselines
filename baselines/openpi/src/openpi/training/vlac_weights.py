import json
import logging
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import re
# Keep import explicit here to avoid NameError if this module is loaded in partial environments.
from openpi.shared import download as _download

LOGGER = logging.getLogger(__name__)

def _extract_episode_and_weight(entry: dict) -> tuple[int | None, float | None]:
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
    try:
        weight = None if weight is None else float(weight)
    except (TypeError, ValueError):
        weight = None

    return episode, weight

def _parse_episode_key(key: str | int) -> int | None:
    # 支持 "episode_00000010" 这类字符串，取末尾数字
    if isinstance(key, int):
        return key
    key = str(key).strip()
    match = re.search(r"(\d+)$", key)
    return int(match.group(1)) if match else None


def _scalar_from_value(value) -> float | None:
    # 支持 list/tuple/ndarray，取第一个能成功转成 float 的元素
    if isinstance(value, (list, tuple, np.ndarray)):
        def _iter(v) -> Iterator:
            if isinstance(v, (list, tuple, np.ndarray)):
                for item in v:
                    yield from _iter(item)
            else:
                yield v

        for item in _iter(value):
            try:
                return float(item)
            except (TypeError, ValueError):
                continue
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_weight_mapping(weight_path: str) -> dict[int, float]:
    local_path = Path(_download.maybe_download(weight_path))
    LOGGER.info("Loading VLAC weights from %s", local_path)

    if local_path.suffix.lower() == ".jsonl":
        entries: Iterable[dict] = (json.loads(line) for line in local_path.read_text().splitlines())
    else:
        entries = json.load(local_path.open("r"))

    mapping: dict[int, float] = {}
    if isinstance(entries, dict):
        for raw_key, raw_value in entries.items():
            episode = _parse_episode_key(raw_key)
            weight = _scalar_from_value(raw_value)
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
                weight = _scalar_from_value(raw_val)
            else:
                weight = _scalar_from_value(weight)
            if episode is None or weight is None:
                continue
            mapping[episode] = weight
    else:
        raise ValueError(f"Unsupported VLAC weight file format: {type(entries)}")

    LOGGER.info("Loaded %d VLAC weights", len(mapping))
    return mapping


def build_weight_table(
    mapping: dict[int, float],
    *,
    default_weight: float = 1.0,
    min_size: int = 0,
) -> np.ndarray | None:
    """Convert a mapping to a dense numpy table suitable for indexing with episode_index."""
    if not mapping and min_size == 0:
        LOGGER.warning("No VLAC weights found and no minimum size requested; weighting will be skipped.")
        return None

    table_size = max(min_size, max(mapping.keys(), default=-1) + 1)
    table = np.full(table_size, default_weight, dtype=np.float32)

    for episode, weight in mapping.items():
        if episode < 0:
            LOGGER.debug("Skipping negative episode index %s in VLAC weights", episode)
            continue
        if episode >= table_size:
            # Extend table if we encounter a larger episode index than anticipated.
            new_table = np.full(episode + 1, default_weight, dtype=np.float32)
            new_table[: table.shape[0]] = table
            table = new_table
            table_size = table.shape[0]
        table[episode] = weight

    return table