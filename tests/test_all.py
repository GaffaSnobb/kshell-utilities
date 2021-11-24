from itertools import zip_longest
import numpy as np
import kshell_utilities.kshell_utilities

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
    res = kshell_utilities.loadtxt(
        path = "summary_test_text_file.txt",
        load_and_save_to_file = False,
        old_or_new = "old"
    )[0]

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

def test_int_vs_floor():
    """
    Check that floor yields the same result as casting to int. This does
    not work for negative values.
    """
    res = kshell_utilities.loadtxt(
        path = "summary_test_text_file.txt",
        load_and_save_to_file = False,
        old_or_new = "old"
    )[0]

    res_1 = np.floor(res.transitions_BM1[:, 5])
    res_2 = res.transitions_BM1[:, 5].astype(int)

    for elem_1, elem_2 in zip(res_1, res_2):
        assert elem_1 == elem_2

def test_file_read_transitions():
    """
    Test that all values in res.transitions are as expected.
    """
    res = kshell_utilities.loadtxt(
        path = "summary_test_text_file.txt",
        load_and_save_to_file = False,
        old_or_new = "old"
    )[0]

    # BE2
    # ---
    spin_initial_expected = np.array([5, 4, 4, 3, 9/2, 9/2])*2
    for i in range(len(res.transitions_BE2[:, 0])):
        msg = "BE2: Error in spin_initial."
        msg += f" Expected: {spin_initial_expected[i]}, got {res.transitions_BE2[i, 0]}."
        assert res.transitions_BE2[i, 0] == spin_initial_expected[i], msg

    parity_initial_expected = np.array([1, 1, 1, 1, -1, -1])
    for i in range(len(res.transitions_BE2[:, 1])):
        msg = "BE2: Error in parity_initial."
        msg += f" Expected: {parity_initial_expected[i]}, got {res.transitions_BE2[i, 1]}."
        assert res.transitions_BE2[i, 1] == parity_initial_expected[i], msg

    Ex_initial_expected = np.array([0.176, 0.182, 0.182, 0.221, 27.281, 27.281])
    for i in range(len(res.transitions_BE2[:, 2])):
        msg = "BE2: Error in Ex_initial."
        msg += f" Expected: {Ex_initial_expected[i]}, got {res.transitions_BE2[i, 2]}."
        assert res.transitions_BE2[i, 2] == Ex_initial_expected[i], msg

    spin_final_expected = np.array([6, 6, 5, 5, 13/2, 5/2])*2
    for i in range(len(res.transitions_BE2[:, 3])):
        msg = "BE2: Error in spin_final."
        msg += f" Expected: {spin_final_expected[i]}, got {res.transitions_BE2[i, 3]}."
        assert res.transitions_BE2[i, 3] == spin_final_expected[i], msg

    parity_final_expected = np.array([1, 1, 1, 1, -1, -1])
    for i in range(len(res.transitions_BE2[:, 4])):
        msg = "BE2: Error in parity_final."
        msg += f" Expected: {parity_final_expected[i]}, got {res.transitions_BE2[i, 4]}."
        assert res.transitions_BE2[i, 4] == parity_final_expected[i], msg

    Ex_final_expected = np.array([0, 0, 0.176, 0.176, 17.937, 18.349])
    for i in range(len(res.transitions_BE2[:, 5])):
        msg = "BE2: Error in Ex_final."
        msg += f" Expected: {Ex_final_expected[i]}, got {res.transitions_BE2[i, 5]}."
        assert res.transitions_BE2[i, 5] == Ex_final_expected[i], msg

    E_gamma_expected = np.array([0.176, 0.182, 0.006, 0.045, 9.344, 8.932])
    for i in range(len(res.transitions_BE2[:, 6])):
        msg = "BE2: Error in E_gamma."
        msg += f" Expected: {E_gamma_expected[i]}, got {res.transitions_BE2[i, 6]}."
        assert res.transitions_BE2[i, 6] == E_gamma_expected[i], msg

    B_decay_expected = np.array([157, 44.8, 3.1, 35, 0, 1.1])
    for i in range(len(res.transitions_BE2[:, 7])):
        msg = "BE2: Error in B_decay."
        msg += f" Expected: {B_decay_expected[i]}, got {res.transitions_BE2[i, 7]}."
        assert res.transitions_BE2[i, 7] == B_decay_expected[i], msg

    B_excite_expected = np.array([132.9, 31, 2.5, 22.3, 0.0, 1.9])
    for i in range(len(res.transitions_BE2[:, 8])):
        msg = "BE2: Error in B_excite."
        msg += f" Expected: {B_excite_expected[i]}, got {res.transitions_BE2[i, 8]}."
        assert res.transitions_BE2[i, 8] == B_excite_expected[i], msg

    # BM1
    # ---
    spin_initial_expected = np.array([2, 2, 1, 2, 7/2, 7/2])*2
    for i in range(len(res.transitions_BM1[:, 0])):
        msg = "BM1: Error in spin_initial."
        msg += f" Expected: {spin_initial_expected[i]}, got {res.transitions_BM1[i, 0]}."
        assert res.transitions_BM1[i, 0] == spin_initial_expected[i], msg

    parity_initial_expected = np.array([1, 1, 1, 1, -1, -1])
    for i in range(len(res.transitions_BM1[:, 1])):
        msg = "BM1: Error in parity_initial."
        msg += f" Expected: {parity_initial_expected[i]}, got {res.transitions_BM1[i, 1]}."
        assert res.transitions_BM1[i, 1] == parity_initial_expected[i], msg

    Ex_initial_expected = np.array([5.172, 17.791, 19.408, 18.393, 24.787, 24.787])
    for i in range(len(res.transitions_BM1[:, 2])):
        msg = "BM1: Error in Ex_initial."
        msg += f" Expected: {Ex_initial_expected[i]}, got {res.transitions_BM1[i, 2]}."
        assert res.transitions_BM1[i, 2] == Ex_initial_expected[i], msg

    spin_final_expected = np.array([0, 0, 2, 2, 5/2, 5/2])*2
    for i in range(len(res.transitions_BM1[:, 3])):
        msg = "BM1: Error in spin_final."
        msg += f" Expected: {spin_final_expected[i]}, got {res.transitions_BM1[i, 3]}."
        assert res.transitions_BM1[i, 3] == spin_final_expected[i], msg

    parity_final_expected = np.array([1, 1, 1, 1, -1, -1])
    for i in range(len(res.transitions_BM1[:, 4])):
        msg = "BM1: Error in parity_final."
        msg += f" Expected: {parity_final_expected[i]}, got {res.transitions_BM1[i, 4]}."
        assert res.transitions_BM1[i, 4] == parity_final_expected[i], msg

    Ex_final_expected = np.array([0, 0, 17.791, 17.791, 21.486, 21.564])
    for i in range(len(res.transitions_BM1[:, 5])):
        msg = "BM1: Error in Ex_final."
        msg += f" Expected: {Ex_final_expected[i]}, got {res.transitions_BM1[i, 5]}."
        assert res.transitions_BM1[i, 5] == Ex_final_expected[i], msg

    E_gamma_expected = np.array([5.172, 17.791, 1.617, 0.602, 3.301, 3.222])
    for i in range(len(res.transitions_BM1[:, 6])):
        msg = "BM1: Error in E_gamma."
        msg += f" Expected: {E_gamma_expected[i]}, got {res.transitions_BM1[i, 6]}."
        assert res.transitions_BM1[i, 6] == E_gamma_expected[i], msg

    B_decay_expected = np.array([20.5, 0.0, 5.7, 0.1, 0.069, 0.463])
    for i in range(len(res.transitions_BM1[:, 7])):
        msg = "BM1: Error in B_decay."
        msg += f" Expected: {B_decay_expected[i]}, got {res.transitions_BM1[i, 7]}."
        assert res.transitions_BM1[i, 7] == B_decay_expected[i], msg

    B_excite_expected = np.array([102.3, 0.0, 3.4, 0.1, 0.092, 0.617])
    for i in range(len(res.transitions_BM1[:, 8])):
        msg = "BM1: Error in B_excite."
        msg += f" Expected: {B_excite_expected[i]}, got {res.transitions_BM1[i, 8]}."
        assert res.transitions_BM1[i, 8] == B_excite_expected[i], msg

if __name__ == "__main__":
    test_file_read_levels()
    test_int_vs_floor()
    test_file_read_transitions()