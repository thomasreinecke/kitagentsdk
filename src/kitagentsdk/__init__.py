# src/kitagentsdk/__init__.py
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("kitagentsdk")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .agent import BaseAgent
from .helpers import run_agent
from .kit import KitClient
