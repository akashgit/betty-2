"""Betty - A peer programming agent for Claude Code."""

try:
    from ._version import __version__
except ImportError:
    try:
        from importlib.metadata import version

        __version__ = version("betty-cli")
    except Exception:
        __version__ = "0.0.0+unknown"
