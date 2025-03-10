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

def calculate_figsize(width, fraction=1):
    """
    Set figure dimensions to avoid scaling in LaTeX.

    https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Parameters
    ----------
    width: float
        Document textwidth or columnwidth in pts
    fraction: float, optional
        Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
        Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    # golden_ratio = (5**.5 - 1) / 2
    ratio = 3/4

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

def abbreviate_string(s, max_length=30):
    if len(s) <= max_length:
        return s
    return f"{s[:max_length//2]}...{s[-max_length//2:]}"

def conditional_red_text(input_string: str, condition: bool) -> str:
    if condition:
        return input_string
    else:
        return f"\033[31m{input_string}\033[0m"