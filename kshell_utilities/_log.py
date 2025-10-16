import logging

from .parameters import flags

class IndentMultilineFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        if "\n" in msg:
            # Split into lines, indent continuation lines
            lines = msg.splitlines()
            header, rest = lines[0], lines[1:]
            indent = " " * (len(self.formatTime(record)) + len(record.levelname) + 4)
            msg = "\n".join([header] + [indent + line for line in rest])
        return msg

logger = logging.getLogger("kshell_utilities")
if not logger.handlers:
    """
    Avoid adding multiple handlers if re-imported.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(IndentMultilineFormatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)