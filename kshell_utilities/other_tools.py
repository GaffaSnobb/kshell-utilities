import sys, os

class HidePrint:
    """
    Simple class for hiding prints to stdout when running unit tests.
    From: https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    Usage:
    ```
    with HidePrint():
        # Code here will not show any prints.

    # Code here will show prints.
    ```
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout