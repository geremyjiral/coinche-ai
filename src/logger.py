import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger."""
    logger = logging.getLogger(name)
    # Prevent duplicate handlers if the logger already exists
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
