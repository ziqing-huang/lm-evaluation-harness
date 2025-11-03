import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import datasets


LOGGER = logging.getLogger(__name__)


def _resolve_data_root(data_root: Optional[str] = None) -> Path:
    """Return the directory containing Uncheatable Eval JSON dumps."""
    candidates: List[Path] = []

    if data_root:
        candidates.append(Path(data_root))

    env_root = os.getenv("UNCHEATABLE_EVAL_DATA_ROOT") or os.getenv(
        "UNCHEATABLE_EVAL_ROOT"
    )
    if env_root:
        candidates.append(Path(env_root))

    current = Path(__file__).resolve()
    for parent in current.parents:
        candidates.append(parent / "uncheatable_eval" / "data")

    candidates.append(Path.cwd() / "uncheatable_eval" / "data")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Unable to locate Uncheatable Eval data directory. Set `data_root` in the task "
        "config or export `UNCHEATABLE_EVAL_DATA_ROOT` to point at the directory that "
        "contains the JSON files such as `wikipedia_english_*.json`."
    )


def load_uncheatable_eval(
    dataset: str,
    data_root: Optional[str] = None,
    max_documents: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
) -> dict:
    """Load Uncheatable Eval documents for lm-evaluation-harness.

    Parameters
    ----------
    dataset:
        Prefix of the dataset files, e.g. ``wikipedia_english``.
    data_root:
        Optional override for the directory that contains the JSON exports.
    max_documents:
        If provided, truncate the dataset to at most this many documents.
    shuffle_seed:
        If provided, shuffle documents deterministically with this seed before truncation.
    """

    root = _resolve_data_root(data_root)
    pattern = f"{dataset}_*.json"
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No Uncheatable Eval files found for prefix '{dataset}' in {root}."
        )

    documents: List[dict] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, list):
            raise ValueError(
                f"Unexpected payload in {file_path}: expected a list of strings."
            )

        for raw_text in payload:
            if not isinstance(raw_text, str):
                raise ValueError(
                    f"Unexpected entry type {type(raw_text)} in {file_path}; "
                    "expected a string."
                )
            text = raw_text.strip()
            if text:
                documents.append({"text": text})

    dataset_obj = datasets.Dataset.from_list(documents)

    if shuffle_seed is not None:
        dataset_obj = dataset_obj.shuffle(seed=shuffle_seed)

    if max_documents is not None:
        max_documents = int(max_documents)
        if max_documents < len(dataset_obj):
            dataset_obj = dataset_obj.select(range(max_documents))

    LOGGER.info(
        "Loaded %d documents for Uncheatable Eval dataset '%s' from %d files in %s",
        len(dataset_obj),
        dataset,
        len(files),
        root,
    )

    return {"test": dataset_obj}


__all__ = ["load_uncheatable_eval"]
