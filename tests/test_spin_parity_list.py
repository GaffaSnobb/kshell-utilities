import kshell_utilities.kshell_utilities
from kshell_utilities.general_utilities import create_spin_parity_list

res = kshell_utilities.loadtxt(
    path = "summary_O19_sdpf-mu.txt",
    load_and_save_to_file = False
)[0]


def test_spin_parity_list():
    """
    Check that all spin-parity pairs appear in spin_parity_list.
    """
    n_transitions = len(res.transitions[:, 0])
    spins = res.levels[:, 1]
    parities = res.levels[:, 2]
    spin_parity_list = create_spin_parity_list(spins, parities)
    index_list = []
    
    for transition_idx in range(n_transitions):
        """
        list.index raises ValueError if x is not in list.
        """
        spin_initial = int(res.transitions[transition_idx, 3])
        parity_initial = int(res.transitions[transition_idx, 1])
        index = spin_parity_list.index([spin_initial, parity_initial])

        if index not in index_list:
            index_list.append(index)

    index_list.sort()
    for i in range(len(index_list)-1):
        msg = "Warning! Not all spin parity pairs have been used!"
        msg += f" {index_list[i+1]} != {index_list[i] + 1}"
        assert index_list[i+1] == index_list[i] + 1, msg
    
if __name__ == "__main__":
    test_spin_parity_list()