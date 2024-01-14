import logging
import sys


LOG_FMT = "[%(asctime)s]::[%(name)s] %(levelname)s - %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level=level)

    if not logger.hasHandlers():
        logger.handlers.clear()

        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(fmt=LOG_FMT, datefmt=LOG_DATEFMT)

        handler.setFormatter(fmt)
        logger.addHandler(handler)
        # https://stackoverflow.com/questions/7173033/duplicate-log-output-when-using-python-logging-module
        logger.propagate = False

    return logger


from . import accept, alns, construct, cvrp, operators

__all__ = (
    "accept",
    "alns",
    "construct",
    "cvrp",
    "operators",
)
