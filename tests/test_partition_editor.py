import os
from kshell_utilities.data_structures import OrbitalParameters
from kshell_utilities.partition_editor import partition_editor
from kshell_utilities.other_tools import HidePrint

def test_partition_editor():
    """
    Add one proton and one neutron configuration to Ni67_gs8_n.ptn with
    a 1hw truncation and check that the new partition file is
    constructed correctly.
    """
    def input_wrapper_test(arg: str):

        if (arg == "Add new proton configuration? (y/n): "):
            return "y"
        
        if (arg == "Add another proton configuration? (y/n): "):
            return "n"
        
        if (arg == "p 0d5/2 (remaining: 20): "):
            return "0"
        
        if (arg == "p 0d3/2 (remaining: 20): "):
            return "0"
        
        if (arg == "p 1s1/2 (remaining: 20): "):
            return "0"
        
        if (arg == "p 0f7/2 (remaining: 20): "):
            return "0"
        
        if (arg == "p 0f5/2 (remaining: 20): "):
            return "6"
        
        if (arg == "p 1p3/2 (remaining: 14): "):
            return "2"
        
        if (arg == "p 1p1/2 (remaining: 12): "):
            return "2"
        
        if (arg == "p 0g9/2 (remaining: 10): "):
            return "2"
        
        if (arg == "p 0g7/2 (remaining: 8): "):
            return "2"
        
        if (arg == "p 1d5/2 (remaining: 6): "):
            return "2"
        
        if (arg == "p 1d3/2 (remaining: 4): "):
            return "2"
        
        if (arg == "p 2s1/2 (remaining: 2): "):
            return "2"
        
        if (arg == "Add new neutron configuration? (y/n): "):
            return "y"
        
        if (arg == "n 0d5/2 (remaining: 31): "):
            return "6"
        
        if (arg == "n 0d3/2 (remaining: 25): "):
            return "4"
        
        if (arg == "n 1s1/2 (remaining: 21): "):
            return "2"
        
        if (arg == "n 0f7/2 (remaining: 19): "):
            return "8"
        
        if (arg == "n 0f5/2 (remaining: 11): "):
            return "6"
        
        if (arg == "n 1p3/2 (remaining: 5): "):
            return "2"
        
        if (arg == "n 1p1/2 (remaining: 3): "):
            return "1"
        
        if (arg == "n 0g9/2 (remaining: 2): "):
            return "2"
        
        if (arg == "n 0g7/2 (remaining: 0): "):
            return "0"
        
        if (arg == "n 1d5/2 (remaining: 0): "):
            return "0"
        
        if (arg == "n 1d3/2 (remaining: 0): "):
            return "0"
        
        if (arg == "n 2s1/2 (remaining: 0): "):
            return "0"
        
        if (arg == "Add another neutron configuration? (y/n): "):
            return "n"

        msg = f"'{arg}' is not accounted for in the testing procedure!"
        raise RuntimeError(msg)

    filename_partition_edited = "tmp_partition_editor_output_can_be_deleted_anytime.ptn"
    filename_partition_original = "Ni67_gs8_n_test.ptn"
    
    partition_editor(
        filename_interaction = "gs8_test.snt",
        filename_partition = filename_partition_original,
        filename_partition_edited = filename_partition_edited,
        input_wrapper = input_wrapper_test,
        is_interactive = False,
    )

    with open(filename_partition_edited, "r") as infile_edited, open(filename_partition_original, "r") as infile_original:
        for i, (line_edited, line_original) in enumerate(zip(infile_edited, infile_original)):
            """
            Loop over the proton configurations, neutron configurations,
            and metadata and check that they are as expected.
            """
            if line_edited != line_original:
                if line_edited == " 87 5\n":
                    """
                    The edited file will have one extra proton and
                    neutron configuration.
                    """
                    continue

                if line_edited == "    87     0  0  0  0  6  2  2  2  2  2  2  2\n":
                    """
                    The edited file will have this as an extra proton
                    configuration. That extra line has its own check in
                    this block.
                    """
                    assert infile_edited.readline() == "# neutron partition\n"
                    continue

                if line_edited == "     5     6  4  2  8  6  2  1  2  0  0  0  0\n":
                    """
                    The edited file will have this as an extra neutron
                    configuration. That extra line has its own check in
                    this block.
                    """
                    assert infile_edited.readline() == "# partition of proton and neutron\n"
                    continue

                if line_edited == "435\n":
                    """
                    The edited file will have more proton-neutron
                    configurations than the original because of the
                    extra proton and neutron configurations.
                    """
                    break

                msg = (
                    f"Error on line {i+1}. {line_edited = }, {line_original = }"
                )
                assert False, msg

        msg = "Incorrect proton-neutron indices: "
        for proton_index in range(1, 87+1):
            """
            Check that the proton-neutron indices are calculated
            correctly.
            """
            for neutron_index in range(1, 5+1):
                calculated = infile_edited.readline().split()
                expected = [str(proton_index), str(neutron_index)]
                assert calculated == expected, f"{msg}{calculated = }, {expected = }"

        msg = (
            "There are still untested lines in the edited file!"
            " It should be exhausted now!"
        )
        assert not infile_edited.readline(), msg

    os.remove(filename_partition_edited)    # Will be removed only if there is no AssertionError.

if __name__ == "__main__":
    with HidePrint():
        test_partition_editor()