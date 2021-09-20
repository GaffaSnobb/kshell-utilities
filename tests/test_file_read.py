import kshell_utilities

res = kshell_utilities.loadtxt("summary_test_text_file.txt")[0]

def test_file_read_excitation_energy():
    """
    Test that kshell_utilities.loadtxt successfully reads excitation
    energy output from KSHELL.

    Raises
    ------
    AssertionError
        If the read values are not exactly equal to the expected values.
    """
    Ex_expected = [0.0, 8.016, 0.240, 0.748, 0.820, 0.894, 1.030, 1.190]

    for calculated, expected in zip(res.Ex, Ex_expected):
        msg = f"Error in E_x. Expected: {expected}, got: {calculated}."
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

    for calculated, expected in zip(res.BE2, BE2_expected):
        msg = f"Error in BE2. Expected: {expected}, got: {calculated}."
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

    for calculated, expected in zip(res.BM1, BM1_expected):
        msg = f"Error in BM1. Expected: {expected}, got: {calculated}."
        success = (calculated[0] == expected[0]) and (calculated[1] == expected[1]) and (calculated[2] == expected[2])
        assert success, msg

if __name__ == "__main__":
    test_file_read_excitation_energy()
    test_file_read_BE2()
    test_file_read_BM1()