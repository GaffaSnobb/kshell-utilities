from typing import Union
import numpy as np

def generate_states(
    start: int = 0,
    stop: int = 14,
    n_states: int = 100,
    parity: Union[str, int] = "both"
    ):
    """
    Generate correct string for input to `kshell_ui.py` when asked for
    which states to calculate. Copy the string generated by this
    function and paste it into `kshell_ui.py` when it prompts for
    states.

    DEPRECATED: RANGE FUNCTIONALITY WAS ADDED IN kshell_ui.py MAKING
    THIS FUNCTION OBSOLETE. WILL BE REMOVED.

    Parameters
    ----------
    start : int
        The lowest spin value.

    stop : int
        The largest spin value.

    n_states : int
        The number of states per spin value.

    parity : Union[str, int]
        The parity of the states. Allowed values are: 1, -1, 'both',
        'positive', 'negative', 'pos', 'neg', '+', '-'.

    Examples
    --------
    ``` python
    >>> import kshell_utilities as ksutil
    >>> ksutil.generate_states(start=0, stop=3, n_states=100, parity="both")
    0+100, 0.5+100, 1+100, 1.5+100, 2+100, 2.5+100, 3+100, 0-100, 0.5-100, 1-100, 1.5-100, 2-100, 2.5-100, 3-100,
    ```
    """
    allowed_positive_parity_inputs = ["positive", "pos", "+", "1", "+1", 1, "both"]
    allowed_negative_parity_inputs = ["negative", "neg", "-", "-1", -1, "both"]
    
    def correct_syntax(lst):
        for elem in lst:
            print(elem, end=", ")
    
    if parity in allowed_positive_parity_inputs:
        positive = [f"{i:g}{'+'}{n_states}" for i in np.arange(start, stop+0.5, 0.5)]
        correct_syntax(positive)
    
    if parity in allowed_negative_parity_inputs:
        negative = [f"{i:g}{'-'}{n_states}" for i in np.arange(start, stop+0.5, 0.5)]
        correct_syntax(negative)