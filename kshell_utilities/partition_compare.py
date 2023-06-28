import multiprocessing, curses
import kshell_utilities as ksutil
from vum import Vum
import numpy as np
from kshell_utilities.data_structures import (
    Partition, Interaction
)
from .partition_tools import _prompt_user_for_interaction_and_partition

def _duplicate_checker(
    partition_a: Partition,
    partition_b: Partition,
) -> np.ndarray[bool]:
    """
    Check if the conffigurations in `partition_a` also occur in
    `partition_b`.
    """
    is_in_both_partitions: np.ndarray[bool] = np.zeros(partition_a.n_configurations, dtype=bool)

    for idx_a in range(partition_a.n_configurations):
        config_a = partition_a.configurations[idx_a].configuration
        
        for idx_b in range(partition_b.n_configurations):
            config_b = partition_b.configurations[idx_b].configuration
            
            if config_a == config_b:
                is_in_both_partitions[idx_a] = True
                break

        else:
            is_in_both_partitions[idx_a] = False

    return is_in_both_partitions

def _sanity_checks(
    filename_partition_a: str,
    filename_partition_b: str,
    filename_interaction: str,
):
    ksutil.test_partition_editor(   # Proof that partition_editor perfectly replicates the combined configs.
        filename_partition = filename_partition_a,
        filename_partition_edited = filename_partition_a + "testfile.ptn",
        filename_interaction = filename_interaction,
    )
    ksutil.test_partition_editor(   # Proof that partition_editor perfectly replicates the combined configs.
        filename_partition = filename_partition_b,
        filename_partition_edited = filename_partition_b + "testfile.ptn",
        filename_interaction = filename_interaction,
    )

def partition_compare():
    """
    Wrapper function for error handling.
    """
    try:
        vum: Vum = Vum()
        msg = _partition_compare(vum)

    except KeyboardInterrupt:
        print("Exiting without saving changes...")

    except curses.error as e:
        raise(e)
    
    except Exception as e:
        raise(e)

    else:
        print(msg)

def _partition_compare(vum: Vum):
    tmp = _prompt_user_for_interaction_and_partition(
        vum = vum,
        is_compare_mode = True,
    )
    filename_interaction, filename_partition_a, filename_partition_b = tmp
    # filename_interaction = "gs8.snt"
    interaction: Interaction = Interaction()
    ksutil.load_interaction(
        filename_interaction = filename_interaction,
        interaction = interaction
    )
    
    # filename_partition_a = "Ni67_gs8_n_edited_2.ptn"
    partition_proton_a: Partition = Partition()
    partition_neutron_a: Partition = Partition()
    partition_combined_a: Partition = Partition()
    ksutil.load_partition(
        filename_partition = filename_partition_a,
        interaction = interaction,
        partition_proton = partition_proton_a,
        partition_neutron = partition_neutron_a,
        partition_combined = partition_combined_a,
    )

    # filename_partition_b = "Ni67_gs8_n.ptn"
    partition_proton_b: Partition = Partition()
    partition_neutron_b: Partition = Partition()
    partition_combined_b: Partition = Partition()
    ksutil.load_partition(
        filename_partition = filename_partition_b,
        interaction = interaction,
        partition_proton = partition_proton_b,
        partition_neutron = partition_neutron_b,
        partition_combined = partition_combined_b,
    )
    
    sanity_check_process = multiprocessing.Process( # Sanity checks in a separate process to make the program more snappy.
        target = _sanity_checks,
        args = (filename_partition_a, filename_partition_b, filename_interaction),
    )
    sanity_check_process.start()
    
    _, mdim_a, _ = ksutil.count_dim(
        model_space_filename = filename_interaction,
        partition_filename = None,
        print_dimensions = False,
        debug = False,
        parity = partition_combined_a.parity,
        proton_partition = [configuration.configuration for configuration in partition_proton_a.configurations],
        neutron_partition = [configuration.configuration for configuration in partition_neutron_a.configurations],
        total_partition = [configuration.configuration for configuration in partition_combined_a.configurations],
    )
    _, mdim_b, _ = ksutil.count_dim(
        model_space_filename = filename_interaction,
        partition_filename = None,
        print_dimensions = False,
        debug = False,
        parity = partition_combined_b.parity,
        proton_partition = [configuration.configuration for configuration in partition_proton_b.configurations],
        neutron_partition = [configuration.configuration for configuration in partition_neutron_b.configurations],
        total_partition = [configuration.configuration for configuration in partition_combined_b.configurations],
    )

    vum.addstr(0, 0,
        string = f"Compare Partitions :: {filename_interaction} :: (A) {filename_partition_a} :: (B) {filename_partition_b}"
    )
    vum.addstr(2, 33,
        f"A"
    )
    vum.addstr(2, 40 + 6,
        f"B",
        is_blank_line = False,
    )
    vum.addstr(3, 0,
        f"dim                   {mdim_a[-1]:14.2e} {mdim_b[-1]:14.2e}"
    )
    vum.addstr(4, 0,
        f"n proton configs      {partition_proton_a.n_configurations:14d} {partition_proton_b.n_configurations:14d}"
    )
    vum.addstr(5, 0,
        f"n neutron configs     {partition_neutron_a.n_configurations:14d} {partition_neutron_b.n_configurations:14d}"
    )
    vum.addstr(6, 0,
        f"n combined configs    {partition_combined_a.n_configurations:14d} {partition_combined_b.n_configurations:14d}"
    )
    
    min_max_a = str((partition_proton_a.ho_quanta_min_this_parity, partition_proton_a.ho_quanta_max_this_parity))[1:-1]#.replace(" ", "")
    min_max_b = str((partition_proton_b.ho_quanta_min_this_parity, partition_proton_b.ho_quanta_max_this_parity))[1:-1]#.replace(" ", "")
    vum.addstr(7, 0,
        f"H.O. min,max proton   {min_max_a:>14s} {min_max_b:>14s}"
    )
    min_max_a = str((partition_neutron_a.ho_quanta_min_this_parity, partition_neutron_a.ho_quanta_max_this_parity))[1:-1]#.replace(" ", "")
    min_max_b = str((partition_neutron_b.ho_quanta_min_this_parity, partition_neutron_b.ho_quanta_max_this_parity))[1:-1]#.replace(" ", "")
    vum.addstr(8, 0,
        f"H.O. min,max neutron  {min_max_a:>14s} {min_max_b:>14s}"
    )
    min_max_a = str((partition_combined_a.ho_quanta_min_this_parity, partition_combined_a.ho_quanta_max_this_parity))[1:-1]#.replace(" ", "")
    min_max_b = str((partition_combined_b.ho_quanta_min_this_parity, partition_combined_b.ho_quanta_max_this_parity))[1:-1]#.replace(" ", "")
    vum.addstr(9, 0,
        f"H.O. min,max combined  {min_max_a:>14s} {min_max_b:>14s}"
    )

    vum.addstr(11, 29,
        f"A in B"
    )
    vum.addstr(11, 36 + 6,
        f"B in A",
        is_blank_line = False,
    )

    a_in_b = _duplicate_checker(
        partition_a = partition_proton_a,
        partition_b = partition_proton_b,
    )
    a_in_b = str(f"{a_in_b.sum()}/{partition_proton_a.n_configurations}")
    b_in_a = _duplicate_checker(
        partition_a = partition_proton_b,
        partition_b = partition_proton_a,
    )
    b_in_a = str(f"{b_in_a.sum()}/{partition_proton_b.n_configurations}")
    vum.addstr(12, 0,
        f"n proton configs    {a_in_b:>14s} {b_in_a:>14s}"
    )
    
    a_in_b = _duplicate_checker(
        partition_a = partition_neutron_a,
        partition_b = partition_neutron_b,
    )
    a_in_b = str(f"{a_in_b.sum()}/{partition_neutron_a.n_configurations}")
    b_in_a = _duplicate_checker(
        partition_a = partition_neutron_b,
        partition_b = partition_neutron_a,
    )
    b_in_a = str(f"{b_in_a.sum()}/{partition_neutron_b.n_configurations}")
    vum.addstr(13, 0,
        f"n neutron configs   {a_in_b:>14s} {b_in_a:>14s}"
    )

    vum.input("Enter any char to exit")

if __name__ == "__main__":
    # main()
    partition_compare()