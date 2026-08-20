"""Shared library for the DSC@UIT 2026 pipeline.

Nothing in here knows about BTC's raw file layout. The only module that does is
``phases/0_harness/ingest.py``, which converts raw BTC files into the canonical
formats documented in the root README. Keep it that way: when BTC ships a data
patch, exactly one file needs editing.
"""
__all__ = [
    "io_utils", "normalize", "chunking", "metrics", "cutoff",
    "bm25", "fusion", "exp_log", "config", "params",
]
