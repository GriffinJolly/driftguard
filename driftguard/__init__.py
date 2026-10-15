"""driftguard: detect silent behavioral drift in deployed LLM APIs."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llm-driftguard")
except PackageNotFoundError:  # running from a source checkout that was never installed
    __version__ = "0.0.0+local"
