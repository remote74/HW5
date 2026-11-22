"""Load the original IMDB splits from a pinned Hugging Face snapshot."""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

DATASET_ID = "stanfordnlp/imdb"
REVISION = "e6281661ce1c48d982bc483cf8a173c1bbeb5d31"
SPLITS = ("train", "test", "unsupervised")
LABELS = {0: "negative", 1: "positive"}


def validate_data(frame: pd.DataFrame, *, supervised: bool = True) -> None:
    """Reject unusable texts or labels before later workflow stages use them."""
    missing = {"text", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("The dataset is empty")
    if frame[["text", "label"]].isna().any().any():
        raise ValueError("Missing text or label values")
    if not frame["text"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("Every text must be a string")
    if frame["text"].str.strip().eq("").any():
        raise ValueError("Empty or whitespace-only texts")
    allowed = set(LABELS) if supervised else {-1}
    if not pd.api.types.is_integer_dtype(frame["label"]):
        raise ValueError("Labels must have an integer dtype")
    if not frame["label"].isin(allowed).all():
        raise ValueError(f"Labels must belong to {sorted(allowed)}")


def load_imdb(cache_dir: str | Path = "data/imdb", *, offline: bool = False,
              include_unsupervised: bool = False) -> dict[str, pd.DataFrame]:
    """Return validated pandas frames, preserving the official split boundaries.

    The unlabeled split is opt-in so later supervised stages cannot accidentally
    treat its -1 sentinel as a sentiment class. Downloads are cached by revision.
    """
    cache = Path(cache_dir) / REVISION
    cache.mkdir(parents=True, exist_ok=True)
    frames = {}
    for split in SPLITS if include_unsupervised else SPLITS[:2]:
        filename = f"{split}-00000-of-00001.parquet"
        path = cache / filename
        if not path.exists():
            if offline:
                raise FileNotFoundError(f"Not cached: {path}. Run once without --offline.")
            url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{REVISION}/plain_text/{filename}"
            temporary = path.with_suffix(".part")
            try:
                urlretrieve(url, temporary)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        frame = pd.read_parquet(path)
        validate_data(frame, supervised=split != "unsupervised")
        frames[split] = frame
    return frames
