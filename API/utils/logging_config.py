import logging
import sys


def setup_logging() -> logging.Logger:
    """Configure a root logger with a simple console handler."""
    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s %(levelname)s [%(name)s:%(module)s:%(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S%z"))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    return root
