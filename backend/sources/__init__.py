"""Per-source scraper modules.

Each module exposes a single entry function that returns a SourceResult so a
failing source is isolated and never crashes the pipeline. The shared shape is
defined in `base.py`.
"""
