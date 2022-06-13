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
    for i in range(len(res.transitions_BE2[:, 3])):
        msg = "BE2: Error in Ex_initial."
        msg += f" Expected: {Ex_initial_expected[i]}, got {res.transitions_BE2[i, 3]}."
        assert res.transitions_BE2[i, 3] == Ex_initial_expected[i], msg

    spin_final_expected = np.array([6, 6, 5, 5, 13/2, 5/2])*2
    for i in range(len(res.transitions_BE2[:, 4])):
        msg = "BE2: Error in spin_final."
        msg += f" Expected: {spin_final_expected[i]}, got {res.transitions_BE2[i, 4]}."
        assert res.transitions_BE2[i, 4] == spin_final_expected[i], msg

    parity_final_expected = np.array([1, 1, 1, 1, -1, -1])
    for i in range(len(res.transitions_BE2[:, 5])):
        msg = "BE2: Error in parity_final."
        msg += f" Expected: {parity_final_expected[i]}, got {res.transitions_BE2[i, 5]}."
        assert res.transitions_BE2[i, 5] == parity_final_expected[i], msg

    Ex_final_expected = np.array([0, 0, 0.176, 0.176, 17.937, 18.349])
    for i in range(len(res.transitions_BE2[:, 7])):
        msg = "BE2: Error in Ex_final."
        msg += f" Expected: {Ex_final_expected[i]}, got {res.transitions_BE2[i, 7]}."
        assert res.transitions_BE2[i, 7] == Ex_final_expected[i], msg

    E_gamma_expected = np.array([0.176, 0.182, 0.006, 0.045, 9.344, 8.932])
    for i in range(len(res.transitions_BE2[:, 8])):
        msg = "BE2: Error in E_gamma."
        msg += f" Expected: {E_gamma_expected[i]}, got {res.transitions_BE2[i, 8]}."
        assert res.transitions_BE2[i, 8] == E_gamma_expected[i], msg

    B_decay_expected = np.array([157, 44.8, 3.1, 35, 0, 1.1])
    for i in range(len(res.transitions_BE2[:, 9])):
        msg = "BE2: Error in B_decay."
        msg += f" Expected: {B_decay_expected[i]}, got {res.transitions_BE2[i, 9]}."
        assert res.transitions_BE2[i, 9] == B_decay_expected[i], msg

    B_excite_expected = np.array([132.9, 31, 2.5, 22.3, 0.0, 1.9])
    for i in range(len(res.transitions_BE2[:, 10])):
        msg = "BE2: Error in B_excite."
        msg += f" Expected: {B_excite_expected[i]}, got {res.transitions_BE2[i, 10]}."
        assert res.transitions_BE2[i, 10] == B_excite_expected[i], msg

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
    for i in range(len(res.transitions_BM1[:, 3])):
        msg = "BM1: Error in Ex_initial."
        msg += f" Expected: {Ex_initial_expected[i]}, got {res.transitions_BM1[i, 3]}."
        assert res.transitions_BM1[i, 3] == Ex_initial_expected[i], msg

    spin_final_expected = np.array([0, 0, 2, 2, 5/2, 5/2])*2
    for i in range(len(res.transitions_BM1[:, 4])):
        msg = "BM1: Error in spin_final."
        msg += f" Expected: {spin_final_expected[i]}, got {res.transitions_BM1[i, 4]}."
        assert res.transitions_BM1[i, 4] == spin_final_expected[i], msg

    parity_final_expected = np.array([1, 1, 1, 1, -1, -1])
    for i in range(len(res.transitions_BM1[:, 5])):
        msg = "BM1: Error in parity_final."
        msg += f" Expected: {parity_final_expected[i]}, got {res.transitions_BM1[i, 5]}."
        assert res.transitions_BM1[i, 5] == parity_final_expected[i], msg

    Ex_final_expected = np.array([0, 0, 17.791, 17.791, 21.486, 21.564])
    for i in range(len(res.transitions_BM1[:, 7])):
        msg = "BM1: Error in Ex_final."
        msg += f" Expected: {Ex_final_expected[i]}, got {res.transitions_BM1[i, 7]}."
        assert res.transitions_BM1[i, 7] == Ex_final_expected[i], msg

    E_gamma_expected = np.array([5.172, 17.791, 1.617, 0.602, 3.301, 3.222])
    for i in range(len(res.transitions_BM1[:, 8])):
        msg = "BM1: Error in E_gamma."
        msg += f" Expected: {E_gamma_expected[i]}, got {res.transitions_BM1[i, 8]}."
        assert res.transitions_BM1[i, 8] == E_gamma_expected[i], msg

    B_decay_expected = np.array([20.5, 0.0, 5.7, 0.1, 0.069, 0.463])
    for i in range(len(res.transitions_BM1[:, 9])):
        msg = "BM1: Error in B_decay."
        msg += f" Expected: {B_decay_expected[i]}, got {res.transitions_BM1[i, 9]}."
        assert res.transitions_BM1[i, 9] == B_decay_expected[i], msg

    B_excite_expected = np.array([102.3, 0.0, 3.4, 0.1, 0.092, 0.617])
    for i in range(len(res.transitions_BM1[:, 10])):
        msg = "BM1: Error in B_excite."
        msg += f" Expected: {B_excite_expected[i]}, got {res.transitions_BM1[i, 10]}."
        assert res.transitions_BM1[i, 10] == B_excite_expected[i], msg

def test_file_read_levels_jem():
    """
    For JEM syntax.

    Raises
    ------
    AssertionError
        If the read values are not exactly equal to the expected values.
    """
    res = kshell_utilities.loadtxt(
        path = "summary_Zn60_jun45_jem_syntax.txt",
        load_and_save_to_file = False,
        old_or_new = "jem"
    )[0]

    E_expected = [
        -50.42584, -49.42999, -47.78510, -46.30908, -46.24604,
        -45.97338, -45.85816, -45.69709, -45.25258, -45.22663
    ]
    spin_expected = [
        0, 4, 8, 4, 0, 4, 12, 2, 8, 6
    ]
    parity_expected = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1
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

def test_file_read_transitions_jem():
    """
    Test that all values in res.transitions are as expected.
    """
    res = kshell_utilities.loadtxt(
        path = "summary_Zn60_jun45_jem_syntax.txt",
        load_and_save_to_file = False,
        old_or_new = "jem"
    )[0]

    E_gs = 50.42584
    
    # BE2
    # ---
    spin_initial_expected = np.array([
        4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10
    ])
    for i in range(len(res.transitions_BE2[:, 0])):
        msg = "BE2: Error in spin_initial."
        msg += f" Expected: {spin_initial_expected[i]}, got {res.transitions_BE2[i, 0]}."
        assert res.transitions_BE2[i, 0] == spin_initial_expected[i], msg

    parity_initial_expected = np.array([
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ])
    for i in range(len(res.transitions_BE2[:, 1])):
        msg = "BE2: Error in parity_initial."
        msg += f" Expected: {parity_initial_expected[i]}, got {res.transitions_BE2[i, 1]}."
        assert res.transitions_BE2[i, 1] == parity_initial_expected[i], msg
    
    Ex_initial_expected = E_gs - np.abs(np.array([
        -42.834, -42.061, -41.992, -41.210, -41.117, -40.802, -40.802, -40.802,
        -40.802, -40.802, -40.802, -40.802, -40.802, -50.426, -50.426, -50.426,
        -50.426, -50.426, -50.426, -50.426, -50.426, -43.879, -43.879, -43.879,
        -43.879, -43.879
    ]))
    for i in range(len(res.transitions_BE2[:, 3])):
        msg = "BE2: Error in Ex_initial."
        msg += f" Expected: {Ex_initial_expected[i]}, got {res.transitions_BE2[i, 3]}."
        assert res.transitions_BE2[i, 3] == Ex_initial_expected[i], msg

    spin_final_expected = np.array([
        0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 10, 10, 10, 10, 10
    ])
    for i in range(len(res.transitions_BE2[:, 4])):
        msg = "BE2: Error in spin_final."
        msg += f" Expected: {spin_final_expected[i]}, got {res.transitions_BE2[i, 4]}."
        assert res.transitions_BE2[i, 4] == spin_final_expected[i], msg

    parity_final_expected = np.array([
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1,
        +1, +1, +1, +1, +1, +1, +1, +1
    ])
    for i in range(len(res.transitions_BE2[:, 5])):
        msg = "BE2: Error in parity_final."
        msg += f" Expected: {parity_final_expected[i]}, got {res.transitions_BE2[i, 5]}."
        assert res.transitions_BE2[i, 5] == parity_final_expected[i], msg

    Ex_final_expected = E_gs - np.abs(np.array([
        -40.802, -40.802, -40.802, -40.802, -40.802, -40.754, -40.666, -40.412,
        -40.375, -40.085, -40.060, -39.988, -39.802, -49.430, -46.309, -45.973,
        -44.988, -44.810, -44.615, -44.470, -43.619, -43.483, -42.618, -42.610,
        -42.173, -42.140
    ]))
    for i in range(len(res.transitions_BE2[:, 7])):
        msg = "BE2: Error in Ex_final."
        msg += f" Expected: {Ex_final_expected[i]}, got {res.transitions_BE2[i, 7]}."
        assert res.transitions_BE2[i, 7] == Ex_final_expected[i], msg

    E_gamma_expected = np.array([
        2.032, 1.259, 1.190, 0.408, 0.315, 0.048, 0.136, 0.390, 0.427, 0.717,
        0.742, 0.814, 1.000, 0.996, 4.117, 4.452, 5.437, 5.616, 5.811, 5.956,
        6.806, 0.396, 1.261, 1.269, 1.706, 1.739
    ])
    for i in range(len(res.transitions_BE2[:, 8])):
        msg = "BE2: Error in E_gamma."
        msg += f" Expected: {E_gamma_expected[i]}, got {res.transitions_BE2[i, 8]}."
        assert res.transitions_BE2[i, 8] == E_gamma_expected[i], msg

    B_decay_expected = np.array([
        0.51104070, 16.07765200, 4.57882840, 8.57092480, 0.46591730,
        0.13763550, 13.32346160, 2.80866440, 266.60020590, 2.26964570,
        1.78483210, 0.08375330, 23.60063140, 675.71309460, 0.32412310,
        0.97507780, 19.98328650, 0.28167560, 5.18973520, 0.05644800,
        0.90882360, 0.07640840, 1.66964180, 0.84907290, 0.28217940, 5.27947770
    ])
    for i in range(len(res.transitions_BE2[:, 9])):
        msg = "BE2: Error in B_decay."
        msg += f" Expected: {B_decay_expected[i]}, got {res.transitions_BE2[i, 9]}."
        assert res.transitions_BE2[i, 9] == B_decay_expected[i], msg

    B_excite_expected = np.array([
        2.55520380, 80.38826020, 22.89414210, 42.85462440, 2.32958690,
        0.02752710, 2.66469230, 0.56173280, 53.32004120, 0.45392910,
        0.35696640, 0.01675060, 4.72012620, 135.14261890, 0.06482460,
        0.19501550, 3.99665730, 0.05633510, 1.03794700, 0.01128960, 0.18176470,
        0.07640840, 1.66964180, 0.84907290, 0.28217940, 5.27947770
    ])
    for i in range(len(res.transitions_BE2[:, 10])):
        msg = "BE2: Error in B_excite."
        msg += f" Expected: {B_excite_expected[i]}, got {res.transitions_BE2[i, 10]}."
        assert res.transitions_BE2[i, 10] == B_excite_expected[i], msg

    # BM1
    # ---
    spin_initial_expected = np.array([
        2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2
    ])
    for i in range(len(res.transitions_BM1[:, 0])):
        msg = "BM1: Error in spin_initial."
        msg += f" Expected: {spin_initial_expected[i]}, got {res.transitions_BM1[i, 0]}."
        assert res.transitions_BM1[i, 0] == spin_initial_expected[i], msg

    parity_initial_expected = np.array([
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1,
        +1, +1, +1
    ])
    for i in range(len(res.transitions_BM1[:, 1])):
        msg = "BM1: Error in parity_initial."
        msg += f" Expected: {parity_initial_expected[i]}, got {res.transitions_BM1[i, 1]}."
        assert res.transitions_BM1[i, 1] == parity_initial_expected[i], msg

    Ex_initial_expected = E_gs -  np.abs(np.array([
        -43.539, -42.609, -40.802, -40.802, -40.802, -40.802, -40.802, -40.802,
        -40.802, -40.802, -40.802, -42.467, -42.467, -42.467, -45.697, -45.001,
        -44.512, -44.439, -44.017, -43.294, -43.063
    ]))
    for i in range(len(res.transitions_BM1[:, 3])):
        msg = "BM1: Error in Ex_initial."
        msg += f" Expected: {Ex_initial_expected[i]}, got {res.transitions_BM1[i, 3]}."
        assert res.transitions_BM1[i, 3] == Ex_initial_expected[i], msg

    spin_final_expected = np.array([
        0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0
    ])
    for i in range(len(res.transitions_BM1[:, 4])):
        msg = "BM1: Error in spin_final."
        msg += f" Expected: {spin_final_expected[i]}, got {res.transitions_BM1[i, 4]}."
        assert res.transitions_BM1[i, 4] == spin_final_expected[i], msg

    parity_final_expected = np.array([
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1,
        +1, +1, +1
    ])
    for i in range(len(res.transitions_BM1[:, 5])):
        msg = "BM1: Error in parity_final."
        msg += f" Expected: {parity_final_expected[i]}, got {res.transitions_BM1[i, 5]}."
        assert res.transitions_BM1[i, 5] == parity_final_expected[i], msg

    Ex_final_expected = E_gs -  np.abs(np.array([
        -40.802, -40.802, -40.786, -40.700, -40.178, -40.068, -39.878, -39.615,
        -39.376, -39.254, -39.167, -35.222, -35.217, -35.174, -42.351, -42.351,
        -42.351, -42.351, -42.351, -42.351, -42.351
    ]))
    for i in range(len(res.transitions_BM1[:, 7])):
        msg = "BM1: Error in Ex_final."
        msg += f" Expected: {Ex_final_expected[i]}, got {res.transitions_BM1[i, 7]}."
        assert res.transitions_BM1[i, 7] == Ex_final_expected[i], msg

    E_gamma_expected = np.array([
        2.737, 1.808, 0.016, 0.101, 0.624, 0.734, 0.924, 1.187, 1.425, 1.548,
        1.635, 7.246, 7.250, 7.293, 3.346, 2.649, 2.161, 2.087, 1.665, 0.943,
        0.711
    ])
    for i in range(len(res.transitions_BM1[:, 8])):
        msg = "BM1: Error in E_gamma."
        msg += f" Expected: {E_gamma_expected[i]}, got {res.transitions_BM1[i, 8]}."
        assert res.transitions_BM1[i, 8] == E_gamma_expected[i], msg

    B_decay_expected = np.array([
        0.00000640, 0.00013110, 0.00023580, 0.00232120, 6.16214940, 0.00008630,
        0.00388720, 0.00055320, 0.00001990, 0.72079000, 0.00458170, 0.00000170,
        0.00142630, 0.00016040, 0.00543100, 0.44305110, 0.00001740, 0.00056840,
        0.00007460, 0.10609890, 0.02403140
    ])
    for i in range(len(res.transitions_BM1[:, 9])):
        msg = "BM1: Error in B_decay."
        msg += f" Expected: {B_decay_expected[i]}, got {res.transitions_BM1[i, 9]}."
        assert res.transitions_BM1[i, 9] == B_decay_expected[i], msg

    B_excite_expected = np.array([
        0.00001940, 0.00039350, 0.00007860, 0.00077370, 2.05404980, 0.00002870,
        0.00129570, 0.00018440, 0.00000660, 0.24026330, 0.00152720, 0.00000050,
        0.00047540, 0.00005340, 0.01629320, 1.32915340, 0.00005220, 0.00170520,
        0.00022400, 0.31829690, 0.07209440
    ])
    for i in range(len(res.transitions_BM1[:, 10])):
        msg = "BM1: Error in B_excite."
        msg += f" Expected: {B_excite_expected[i]}, got {res.transitions_BM1[i, 10]}."
        assert res.transitions_BM1[i, 10] == B_excite_expected[i], msg

if __name__ == "__main__":
    test_file_read_levels()
    test_int_vs_floor()
    test_file_read_transitions()
    test_file_read_levels_jem()
    test_file_read_transitions_jem()