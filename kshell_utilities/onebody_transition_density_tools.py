import numpy.typing as npt
from collections.abc import KeysView

def get_included_transitions_obtd_dict_keys(
    included_transitions: npt.NDArray,
    obtd_dict_keys: KeysView[tuple[int, ...]]
) -> list[tuple[int, ...]]:
    """
    Calculate which OBTD dict keys are present in the array of included
    transitions.

    Parameters
    ----------
    included_transitions : npt.NDArray
        The transitions which were used to calculate the GSF in some excitation
        energy interval.

    obtd_dict_keys : KeysView[tuple[int, ...]]
        The keys of the OBTD dict.

    Returns
    -------
    included_transitions_keys : list[tuple[int, ...]]
        A list of OBTD keys which corresponds to transitions which are in
        `included_transitions`.
    """
    included_transitions_keys: list[tuple[int, ...]] = []
    obtd_skips: set[tuple[int, ...]] = set()

    for transition_idx in range(len(included_transitions)):
        j_i, pi_i, idx_i, Ex_i, j_f, pi_f, idx_f, Ex_f, E_gamma, B_if, B_fi = included_transitions[transition_idx]
        j_i   = int(j_i)    # int casts are not very important, just for more clear printing.
        pi_i  = int(pi_i)
        idx_i = int(idx_i)
        j_f   = int(j_f)
        pi_f  = int(pi_f)
        idx_f = int(idx_f)
        master_key = (j_i, pi_i, j_f, pi_f)
        key = (j_i, pi_i, idx_i, j_f, pi_f, idx_f)  # Keys for the OBTD dict.
        
        if master_key not in obtd_dict_keys:
            """
            There might not exist OBTDs for all possible transitions.
            If the master key does not exist, any keys with the same
            (j_i, pi_i, j_f, pi_f) should not exist either.
            """
            obtd_skips.add(master_key)
            continue
        
        included_transitions_keys.append(key)

    assert len(included_transitions_keys) == len(set(included_transitions_keys)), "Duplicate keys detected! Each key should only appear once!"

    print(f"\nCould not find OBTDs for the following (j_i, pi_i, j_f, pi_f) in the given gamma energy range:")
    for skip in obtd_skips:
        print(skip)
    print()

    return included_transitions_keys