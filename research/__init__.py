# research/__init__.py

# WHAT: Import the main public API function.
# WHY: This exposes `run_deep_research` at the top level of the `research` package.
# It allows external modules (like a server ) to import it cleanly
# with `from research import run_deep_research`, rather than needing to know the
# internal file structure (`from research.pipeline import run_deep_research`).
from .pipeline import run_deep_research