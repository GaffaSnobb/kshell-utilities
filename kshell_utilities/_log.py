import logging

from .parameters import flags

logger = logging.getLogger("kshell_utilities")
if not logger.handlers:
    """
    Avoid adding multiple handlers if re-imported.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)