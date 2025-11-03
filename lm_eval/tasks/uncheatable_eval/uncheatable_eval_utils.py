import gzip
import json
import logging
import os
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional

import datasets


LOGGER = logging.getLogger(__name__)

TEXT_FIELD_CANDIDATES: tuple[str, ...] = (
    "text",
    "body",
    "content",
    "article",
    "document",
    "raw_text",
    "code",
    "message",
    "description",
    "story",
)

LIST_FIELD_CANDIDATES: tuple[str, ...] = (
    "paragraphs",
    "sentences",
    "lines",
    "messages",
)


def _resolve_data_root(data_root: Optional[str] = None) -> Path:
    """Return the directory containing Uncheatable Eval dumps."""

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
        candidates.extend(
            [
                parent / "uncheatable_eval" / "data",
                parent / "uncheatable-eval" / "data",
                parent / "raw" / "uncheatable-eval" / "latest",
                parent / "raw" / "uncheatable_eval" / "latest",
                parent / "local_store" / "raw" / "uncheatable-eval" / "latest",
                parent / "local_store" / "raw" / "uncheatable_eval" / "latest",
            ]
        )

    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / "uncheatable_eval" / "data",
            cwd / "uncheatable-eval" / "data",
            cwd / "raw" / "uncheatable-eval" / "latest",
            cwd / "raw" / "uncheatable_eval" / "latest",
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Unable to locate Uncheatable Eval data directory. Set `data_root` in the task "
        "config or export `UNCHEATABLE_EVAL_DATA_ROOT` to point at the directory that "
        "contains the normalized JSONL files such as `wikipedia_english_*.jsonl.gz`."
    )


def load_uncheatable_eval(
    dataset: str,
    data_root: Optional[str] = None,
    max_documents: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
    **_
) -> dict:
    """Load Uncheatable Eval documents for lm-evaluation-harness."""

    root = _resolve_data_root(data_root)
    patterns = [
        f"{dataset}_*.jsonl.gz",
        f"{dataset}_*.jsonl",
        f"{dataset}_*.json",
    ]
    files = list(
        chain.from_iterable(sorted(root.glob(pattern)) for pattern in patterns)
    )
    if not files:
        raise FileNotFoundError(
            f"No Uncheatable Eval files found for prefix '{dataset}' in {root}."
        )

    records = list(_iter_dataset_records(files))
    if not records:
        raise ValueError(
            f"No usable records found for Uncheatable Eval dataset '{dataset}' in {root}."
        )

    dataset_obj = datasets.Dataset.from_list(records)

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


def _iter_dataset_records(files: Iterable[Path]) -> Iterator[dict[str, str]]:
    for file_path in files:
        if file_path.name.endswith(".jsonl.gz"):
            yield from _iter_jsonl(file_path, compression="gzip")
        elif file_path.suffix == ".jsonl":
            yield from _iter_jsonl(file_path)
        elif file_path.suffix == ".json":
            yield from _iter_json(file_path)
        else:
            LOGGER.warning("Skipping unsupported file %s", file_path)


def _iter_jsonl(file_path: Path, compression: Optional[str] = None) -> Iterator[dict[str, str]]:
    opener = gzip.open if compression == "gzip" else open
    with opener(file_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            yield _normalize_record(raw, file_path)


def _iter_json(file_path: Path) -> Iterator[dict[str, str]]:
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(
            f"Unexpected payload in {file_path}: expected a list, found {type(payload).__name__}."
        )

    for raw in payload:
        yield _normalize_record(raw, file_path)


def _normalize_record(raw: Any, file_path: Path) -> dict[str, str]:
    text = _extract_text(raw)
    if text is None or not str(text).strip():
        raise ValueError(f"Record in {file_path} does not contain text.")
    return {"text": str(text)}


def _extract_text(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        value = raw.get("text")
        if isinstance(value, str) and value.strip():
            return value
        for key in TEXT_FIELD_CANDIDATES:
            candidate = raw.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        for key in TEXT_FIELD_CANDIDATES:
            joined = _join_list_field(raw.get(key))
            if joined:
                return joined
        for key in LIST_FIELD_CANDIDATES:
            joined = _join_list_field(raw.get(key))
            if joined:
                return joined
        title = raw.get("title")
        body = raw.get("body")
        if isinstance(title, str) and isinstance(body, str):
            combined = f"{title.strip()}\n\n{body.strip()}".strip()
            if combined:
                return combined
        if isinstance(title, str) and title.strip():
            return title
        return json.dumps(raw, ensure_ascii=False)
    return str(raw)


def _join_list_field(value: Any) -> Optional[str]:
    if isinstance(value, list):
        text_items = [str(item) for item in value if item is not None]
        if text_items:
            return "\n".join(text_items)
    return None


__all__ = ["load_uncheatable_eval"]
