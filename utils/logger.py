import logging
import sys
import time
from typing import Callable, Optional

# Configure standard logger
logger = logging.getLogger("ai_video_assistant")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class PipelineTimer:
    """
    Context manager and tracker for timing pipeline stages.
    """
    def __init__(self, stage_name: str, callback: Optional[Callable[[str, float, str], None]] = None):
        self.stage_name = stage_name
        self.callback = callback
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        logger.info(f"Starting [{self.stage_name}]...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start_time
        if exc_type is None:
            logger.info(f"Completed [{self.stage_name}] in {self.elapsed:.2f}s")
        else:
            logger.error(f"Failed [{self.stage_name}] after {self.elapsed:.2f}s: {exc_val}")
        return False
