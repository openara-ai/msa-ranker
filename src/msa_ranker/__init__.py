"""msa-ranker — learned (non-LLM) learning-to-rank reranker for media-search-agent."""

from importlib.metadata import PackageNotFoundError, version

try:
    # Resolved from the installed distribution metadata, which setuptools_scm
    # derives from the git tag — single source of truth (no hand-pinned string).
    __version__ = version("msa-ranker")
except PackageNotFoundError:  # running from a source tree before install
    __version__ = "0.0.0"
