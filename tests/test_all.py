from itertools import zip_longest
import numpy as np
import kshell_utilities.kshell_utilities

res = kshell_utilities.loadtxt(
    path = "summary_test_text_file.txt",
    load_and_save_to_file = False
)[0]

def test_file_read_levels():
    """
    Test that kshell_utilities.loadtxt successfully reads excitation
    energy output from KSHELL. Note that -1 spin states are supposed to
    be skipped.

    Raises
    ------
    AssertionError
        If the read values are not exactly equal to the expected values.
    """
    E_expected = [
        -41.394, -33.378, -114.552, -114.044, -113.972, -113.898, -113.762,
        -113.602, -9.896, -9.052
    ]
    spin_expected = [
        2*0, 2*3, 2*2, 2*5, 2*4, 2*7, 2*2, 2*3, 2*3/2, 2*7/2
    ]
    parity_expected = [
        1, 1, 1, 1, 1, 1, 1, 1, -1, -1
    ]

    for calculated, expected in zip_longest(res.levels[:, 0], E_expected):
        msg = f"Error in Ex. Expected: {expected}, got: {calculated}."
        assert calculated == expected, msg

    for calculated, expected in zip_longest(res.levels[:, 1], spin_expected):
        msg = f"Error in spin. Expected: {expected}, got: {calculated}."
        assert calculated == expected, msg

    for calculated, expected in zip_longest(res.levels[:, 2], parity_expected):
        msg = f"Error in parity. Expected: {expected}, got: {calculated}."
        assert calculated == expected, msg

def test_file_read_BE2():
    """
    Test that kshell_utilities.loadtxt successfully reads BE2 output
    from KSHELL.

    Raises
    ------
    AssertionError
        If the read values are not exactly equal to the expected values.
    """

    BE2_expected = [
        [0.176, 157.0, 0.176],
        [0.182,  44.8, 0.182],
        [0.182,   3.1, 0.006],
        [0.221,  35.0, 0.045]
    ]

    for calculated, expected in zip_longest(res.BE2, BE2_expected, fillvalue=[None, None, None]):
        msg = f"Error in BE2. Expected: {expected}, got: {calculated}."
        
        if len(BE2_expected) != len(res.BE2):
            msg += f" Check for -1 spin states."
        
        success = (calculated[0] == expected[0]) and (calculated[1] == expected[1]) and (calculated[2] == expected[2])
        assert success, msg

def test_file_read_BM1():
    """
    Test that kshell_utilities.loadtxt successfully reads BM1 output
    from KSHELL.

    Raises
    ------
    AssertionError
        If the read values are not exactly equal to the expected values.
    """

    BM1_expected = [
        [5.172, 20.5, 5.172],
        [17.791, 0.0, 17.791],
        [19.408, 5.7, 1.617],
        [18.393, 0.1, 0.602]
    ]

    for calculated, expected in zip_longest(res.BM1, BM1_expected, fillvalue=[None, None, None]):
        msg = f"Error in BM1. Expected: {expected}, got: {calculated}."

        if len(BM1_expected) != len(res.BM1):
            msg += f" Check for -1 spin states."

        success = (calculated[0] == expected[0]) and (calculated[1] == expected[1]) and (calculated[2] == expected[2])
        assert success, msg

def test_int_vs_floor():
    """
    Check that floor yields the same result as casting to int. This does
    not work for negative values.
    """
    res_1 = np.floor(res.transitions[:, 5])
    res_2 = res.transitions[:, 5].astype(int)

    for elem_1, elem_2 in zip(res_1, res_2):
        assert elem_1 == elem_2

def test_initial_and_final_parity():
    pass

if __name__ == "__main__":
    test_file_read_levels()
    test_file_read_BE2()
    test_file_read_BM1()
    test_int_vs_floor()
    test_initial_and_final_parity()