import time, os, curses
from typing import Callable
from vum import Vum
import numpy as np
from .count_dim import count_dim
from .kshell_exceptions import KshellDataStructureError
from .parameters import (
    spectroscopic_conversion, shell_model_order, major_shell_order
)
from .data_structures import (
    OrbitalParameters, Configuration, ModelSpace, Interaction, Partition
)

DELAY: int = 2  # Delay time for time.sleep(DELAY) in seconds

class ScreenDummy:
    def clear(self):
        return

class VumDummy:
    def __init__(self) -> None:
        self.screen = ScreenDummy()
        self.command_log_length = 0
        self.command_log_length = 0
        self.n_rows = 0
        self.n_cols = 0

    def addstr(self, y, x, text, is_blank_line=None):
        return

    def input(self, _) -> str:
        return "n"

def _calculate_configuration_parity(
    configuration: list[int],
    model_space: list[OrbitalParameters]
) -> int:
    """
    Calculate the parity of a configuration.

    Parameters
    ----------
    configuration : list[int]
        The configuration to calculate the parity of.
    
    model_space : list[OrbitalParameters]
        The model space orbitals to use for the calculation.
    """
    if not configuration:
        msg = "Configuration is empty! Undefined behaviour."
        raise ValueError(msg)

    parity: int = 1
    for i in range(len(configuration)):
        if not configuration[i]: continue   # Empty orbitals do not count towards the total parity.
        parity *= model_space[i].parity**configuration[i]

    return parity

def _analyse_existing_configuration(
    vum: Vum,
    proton_configurations: list[Configuration],
    neutron_configurations: list[Configuration],
    input_wrapper: Callable,
    model_space: list[OrbitalParameters],
    y_offset: int,
    n_proton_orbitals: int,
) -> None:
    """
    Prompt the user for an index of an existing configuration and show
    an analysis of the configuration in question. This function is only
    used once and exists simply to make the amount of code lines in
    `_partition_editor` smaller.

    Parameters
    ----------
    vum : Vum
        The Vum object to draw the map with.
    
    proton_configurations : list[Configuration]
        The proton configurations to choose from.

    neutron_configurations : list[Configuration]
        The neutron configurations to choose from.

    input_wrapper : Callable
        The input wrapper to use for user input.

    model_space : list[OrbitalParameters]
        The model space orbitals to use for the calculation.

    y_offset : int
        The y offset to use for drawing the map.
    """
    pn_configuration_dict: dict[str, list[Configuration]] = {
        "p": proton_configurations,
        "n": neutron_configurations,
    }
    while True:
        if input_wrapper("Analyse existing configuration? (y/n)") == "y":
            while True:
                p_or_n = input_wrapper("Proton or neutron configuration? (p/n)")
                if (p_or_n == "p"):
                    is_proton = True
                    is_neutron = False
                    orbital_idx_offset: int = 0
                    break
                if (p_or_n == "n"):
                    is_proton = False
                    is_neutron = True
                    orbital_idx_offset: int = n_proton_orbitals # This is needed to index the neutron orbitals correctly when drawing the map.
                    break
            
            draw_shell_map(vum=vum, model_space=model_space, is_proton=is_proton, is_neutron=is_neutron)
            
            while True:
                configuration_idx = input_wrapper(f"Choose a {p_or_n} configuration index in 1, 2, ..., {len(pn_configuration_dict[p_or_n])} (q to quit)")
                if configuration_idx == "q": break
                
                try:
                    configuration_idx = int(configuration_idx)
                except ValueError:
                    draw_shell_map(vum=vum, model_space=model_space, is_proton=is_proton, is_neutron=is_neutron)
                    continue
                
                configuration_idx -= 1    # List indices are from 0 while indices in .ptn are from 1.
                if configuration_idx in range(len(pn_configuration_dict[p_or_n])):

                    current_configuration = pn_configuration_dict[p_or_n][configuration_idx].configuration
                    configuration_parity = 1
                    for i in range(len(current_configuration)):
                        configuration_parity *= model_space[i].parity**current_configuration[i] if current_configuration[i] else 1  # Avoid multiplication by 0.
                        draw_shell_map(
                            vum = vum,
                            model_space = model_space,
                            is_proton = is_proton,
                            is_neutron = is_neutron,
                            occupation = (orbital_idx_offset + i, current_configuration[i]),
                        )

                    vum.addstr(y_offset + 8, 0, f"parity current configuration = {configuration_parity}")
                
                else:
                    draw_shell_map(vum=vum, model_space=model_space, is_proton=is_proton, is_neutron=is_neutron)

        else: break # If answer from user is not 'y'.

    vum.addstr(y_offset + 8, 0, "parity current configuration = None")
    draw_shell_map(vum=vum, model_space=model_space, is_proton=True, is_neutron=True)

def draw_shell_map(
    vum: Vum,
    model_space: list[OrbitalParameters],
    is_proton: bool,
    is_neutron: bool,
    occupation: tuple[int, int] | None = None,
    n_holes: int = 0,
) -> None:
    """
    Draw a simple map of the model space orbitals of the current
    interaction file. Sort the orbitals based on the shell_model_order
    dict.

    Parameters
    ----------
    vum : Vum
        The Vum object to draw the map with.

    model_space : list[OrbitalParameters]
        The model space orbitals to draw.

    is_proton : bool
        Whether to draw the proton map.

    is_neutron : bool
        Whether to draw the neutron map.

    occupation : tuple[int, int] | None, optional
        The occupation of the model space orbitals. The tuple contains
        (orbital index, occupation). If None, draw the entire map with
        no occupation, by default None.

    n_holes : int
        For indicating the original position of excited particles.

    Returns
    -------
    None
    """
    y_offset: int = 11
    model_space_copy = sorted(  # Sort the orbitals based on the shell_model_order dict.
        model_space,
        key = lambda orbital: shell_model_order[f"{orbital.n}{spectroscopic_conversion[orbital.l]}{orbital.j}"].idx,
        reverse = True
    )
    model_space_proton = [orbital for orbital in model_space_copy if orbital.tz == -1]
    model_space_neutron = [orbital for orbital in model_space_copy if orbital.tz == 1]

    max_neutron_j: int = max([orbital.j for orbital in model_space_neutron], default=0)
    max_proton_j: int = max([orbital.j for orbital in model_space_proton], default=0)  # Use the max j value to center the drawn orbitals.

    if is_proton:
        proton_offset: int = 17 + (max_proton_j + 1)*2  # The offset for the neutron map.
        is_blank_line: bool = False
        if occupation is None:
            """
            Draw the entire map with no occupation.
            """
            for i in range(len(model_space_proton)):
                string = (
                    f"{model_space_proton[i].idx + 1:2d}"
                    f" {model_space_proton[i].name}"
                    f" {model_space_proton[i].parity:2d} " +
                    " "*(max_proton_j - model_space_proton[i].j) + "-" + " -"*(model_space_proton[i].j + 1)
                )
                vum.addstr(i + y_offset, 0, string)
        else:
            """
            Re-draw a single orbital with the user inputted occupation.
            """
            location: int = 0
            for orbital in model_space_proton:
                if orbital.idx == occupation[0]: break
                location += 1
            else:
                msg = (
                    f"Orbital index {occupation[0]} not found in the"
                    " current model space!"
                )
                raise RuntimeError(msg)
            
            string = (
                f"{orbital.idx + 1:2d}"
                f" {orbital.name}"
                f" {orbital.parity:2d} " +
                " "*(max_proton_j - orbital.j) + "-"
            )
            if not n_holes:
                string += "o-"*occupation[1] + " -"*(orbital.j + 1 - occupation[1])
            else:
                string += "o-"*(occupation[1] - n_holes) + "x-"*n_holes + " -"*(orbital.j + 1 - occupation[1])
            vum.addstr(location + y_offset, 0, string)

    else:
        proton_offset: int = 0
        is_blank_line: bool = True

    if is_neutron:
        if occupation is None:
            for i in range(len(model_space_neutron)):        
                string = (
                    f"{model_space_neutron[i].idx + 1:2d}"
                    f" {model_space_neutron[i].name}"
                    f" {model_space_neutron[i].parity:2d} " +
                    " "*(max_neutron_j - model_space_neutron[i].j) + "-" + " -"*(model_space_neutron[i].j + 1)
                )
                vum.addstr(i + y_offset, proton_offset, string, is_blank_line=is_blank_line)
        
        else:
            """
            Re-draw a single orbital with the user inputted occupation.
            """
            location: int = 0
            for orbital in model_space_neutron:
                if orbital.idx == occupation[0]: break
                location += 1
            else:
                msg = (
                    f"Orbital index {occupation[0]} not found in the"
                    " current model space!"
                )
                raise RuntimeError(msg)
            
            string = (
                f"{orbital.idx + 1:2d}"
                f" {orbital.name}"
                f" {orbital.parity:2d} " +
                " "*(max_neutron_j - orbital.j) + "-"
            )
            string += "o-"*occupation[1] + " -"*(orbital.j + 1 - occupation[1])
            vum.addstr(location + y_offset, proton_offset, string)

def _generate_total_configurations(
    partition_proton: Partition,
    partition_neutron: Partition,
    partition_combined: Partition,
    partition_file_parity: int,
) -> None:
    """
    Generate all the possible combinations of proton and neutron
    configurations. The parities of the proton and neutron
    configurations must combine multiplicatively to the parity of
    the partition file. The `combined_configurations` list will be
    cleared before the new configurations are added.

    Parameters
    ----------
    partition_file_parity : int
        The parity of the partition file.

    Returns
    -------
    None

    Raises
    ------
    KshellDataStructureError
        If any proton or neutron configuration is unused. This happens
        if the parity of the configuration does not multiplicatively
        combine with any of the configurations of the opposite nucleon
        to the parity of the partition file.
    """
    ho_quanta_min_before = partition_combined.ho_quanta_min
    ho_quanta_max_before = partition_combined.ho_quanta_max
    print(f"{ho_quanta_min_before = }")
    print(f"{ho_quanta_max_before = }")
    HO_MIN_TMP: int = 60
    HO_MAX_TMP: int = 63
    partition_combined.clear()
    neutron_configurations_count: list[int] = [0]*partition_neutron.n_configurations
    proton_configurations_count: list[int] = [0]*partition_proton.n_configurations
    
    for p_idx in range(partition_proton.n_configurations):
        for n_idx in range(partition_neutron.n_configurations):
            parity_tmp: int = partition_proton.configurations[p_idx].parity*partition_neutron.configurations[n_idx].parity
            if parity_tmp != partition_file_parity:
                """
                Only combinations of proton and neutron orbitals with
                the same parity as the parity of the partition file are
                accepted.

                NOTE: !!! DET ER HER FEILEN ER!! MÅ TENKE PÅ HVOR MANGE
                HW SOM ER TILLATT NÅR PN-KOMBINASJONENE REGNES UT!!
                """
                continue

            ho_quanta_tmp: int = (
                partition_proton.configurations[p_idx].ho_quanta + 
                partition_neutron.configurations[n_idx].ho_quanta
            )
            if not (HO_MIN_TMP <= ho_quanta_tmp <= HO_MAX_TMP):
                """
                The combined harmonic oscillator quanta of the combined
                proton and neutron orbitals must be within the initial
                limits.
                """
                continue

            neutron_configurations_count[n_idx] += 1
            proton_configurations_count[p_idx] += 1

            if partition_file_parity == -1: partition_combined.n_new_negative_configurations += 1
            if partition_file_parity == +1: partition_combined.n_new_positive_configurations += 1

            partition_combined.ho_quanta_min = min(partition_combined.ho_quanta_min, ho_quanta_tmp)
            partition_combined.ho_quanta_max = max(partition_combined.ho_quanta_max, ho_quanta_tmp)

            partition_combined.configurations.append(
                Configuration(
                    configuration = [p_idx, n_idx],
                    parity = parity_tmp,
                    ho_quanta = ho_quanta_tmp,
                )
            )

    # for p_idx in range(partition_proton.n_configurations):
    #     if proton_configurations_count[p_idx] == 0:
    #         msg = (
    #             f"Proton configuration {p_idx} ({partition_proton.configurations[p_idx].configuration}) has not been paired with"
    #             " any neutron configuration due to parity mismatch."
    #             f" Unable to produce partition file parity ({partition_file_parity})"
    #             f" with the parity of this configuration ({partition_proton.configurations[p_idx].parity})."
    #         )
    #         raise KshellDataStructureError(msg)

    # for n_idx in range(partition_neutron.n_configurations):
    #     if neutron_configurations_count[n_idx] == 0:
    #         msg = (
    #             f"Neutron configuration {n_idx} ({partition_neutron.configurations[n_idx].configuration}) has not been paired with"
    #             " any proton configuration due to parity mismatch."
    #             f" Unable to produce partition file parity ({partition_file_parity})"
    #             f" with the parity of this configuration ({partition_neutron.configurations[n_idx].parity})."
    #         )
    #         raise KshellDataStructureError(msg)

    assert ho_quanta_min_before == partition_combined.ho_quanta_min
    assert ho_quanta_max_before == partition_combined.ho_quanta_max

def _check_configuration_duplicate(
    new_configuration: list[int],
    existing_configurations: list[Configuration],
) -> bool | list[int]:
    
    for i, configuration in enumerate(existing_configurations):
        if new_configuration == configuration.configuration:
            return [str(i), configuration]
        
    return False

def _add_npnh_excitations(
    vum: Vum,
    input_wrapper: Callable,
    model_space_slice: ModelSpace,
    interaction: Interaction,
    partition: Partition,
    nucleon_choice: str,
    is_proton: bool,
    is_neutron: bool,
) -> bool:
    while True:
        """
        Prompt user for the number of particles to excite.
        """
        n_particles_choice = input_wrapper("N-particle N-hole (N)")
        if n_particles_choice == "q": break
        
        try:
            n_particles_choice = int(n_particles_choice)
        except ValueError:
            continue

        if n_particles_choice != 2:
            vum.addstr(
                vum.n_rows - 3 - vum.command_log_length, 0,
                "INVALID: Currently hard-coded for 2 particle excitation!"
            )
            continue

        if n_particles_choice < 1:
            vum.addstr(
                vum.n_rows - 3 - vum.command_log_length, 0,
                "INVALID: The number of particles must be larger than 0."
            )
            continue

        break
    
    if n_particles_choice == "q": return False

    while True:
        """
        Prompt user for the number of excitations to add.
        """
        n_excitations_choice = input_wrapper("How many excitations to add? (amount/all)")
        if n_excitations_choice == "q": break
        
        try:
            n_excitations_choice = int(n_excitations_choice)
        except ValueError:
            continue

        if n_excitations_choice < 1:
            vum.addstr(
                vum.n_rows - 3 - vum.command_log_length, 0,
                "INVALID: The number of excitations must be larger than 0."
            )
            continue

        break
    
    if n_excitations_choice == "q": return False

    while True:
        """
        Prompt the user for which major shells to include
        in the N-particle N-hole excitation.
        """
        initial_major_shell_choice = input_wrapper(f"Initial major shell? ({model_space_slice.major_shell_names})")
        if initial_major_shell_choice == "q": break

        if initial_major_shell_choice in model_space_slice.major_shell_names: break
    
    if initial_major_shell_choice == "q": return False

    while True:
        """
        Prompt the user for which major shells to include
        in the N-particle N-hole excitation.
        """
        final_major_shell_choice = input_wrapper(f"Final major shell? ({model_space_slice.major_shell_names})")
        if final_major_shell_choice == "q": break

        if final_major_shell_choice in model_space_slice.major_shell_names: break

    if final_major_shell_choice == "q": return False
    
    if initial_major_shell_choice == final_major_shell_choice:
        vum.addstr(
            vum.n_rows - 3 - vum.command_log_length, 0,
            "INVALID: Initial and final major shell cannot be the same!"
        )
        return False

    if major_shell_order[initial_major_shell_choice] > major_shell_order[final_major_shell_choice]:
        vum.addstr(
            vum.n_rows - 3 - vum.command_log_length, 0,
            "INVALID: Initial major shell cannot be higher energy than final major shell!"
        )
        return False

    vum.addstr( # Remove any error messages.
        vum.n_rows - 3 - vum.command_log_length, 0, " "
    )
    initial_orbital_indices: list[int] = [] # Store the indices of the orbitals which are in the initial major shell.
    final_orbital_indices: list[int] = [] # Store the indices of the orbitals which are in the final major shell.
    final_orbital_degeneracy: dict[int, int] = {}    # Accompanying degeneracy of the orbital.
    new_configurations: list[Configuration] = []    # Will be merged with partition.configurations at the end of this function.

    for orb in model_space_slice.orbitals:
        """
        Extract indices and degeneracies of the initial and
        final orbitals.
        """
        if orb.order.major_shell_name == initial_major_shell_choice:
            if is_proton:
                init_orb_idx = orb.idx
            elif is_neutron:
                init_orb_idx = orb.idx - interaction.model_space_proton.n_orbitals
            else:
                msg = "'is_proton' and 'is_neutron' should never both be True or False at the same time!"
                raise ValueError(msg)
            
            initial_orbital_indices.append(init_orb_idx)
        
        elif orb.order.major_shell_name == final_major_shell_choice:
            if is_proton:
                final_orb_idx = orb.idx
            elif is_neutron:
                final_orb_idx = orb.idx - interaction.model_space_proton.n_orbitals

            final_orbital_indices.append(final_orb_idx)
            final_orbital_degeneracy[final_orb_idx] = orb.j + 1    # j is stored as j*2.

    for configuration in partition.configurations:
        """
        Loop over every existing configuration.
        """
        for init_orb_idx in initial_orbital_indices:
            """
            Case: N particles from from the same initial
            orbital are excited to the same final orbital.
            """
            if configuration.configuration[init_orb_idx] < n_particles_choice: continue   # Cannot excite enough particles.
            
            for final_orb_idx in final_orbital_indices:
                """
                Both particles to the same final orbital.
                """
                max_additional_occupation = final_orbital_degeneracy[final_orb_idx] - configuration.configuration[final_orb_idx]
                assert max_additional_occupation >= 0, "'max_additional_occupation' should never be negative!"  # Development sanity test.
                
                if max_additional_occupation < n_particles_choice: continue # Cannot excite enough particles.
                
                new_configuration = configuration.configuration.copy()
                new_configuration[init_orb_idx] -= n_particles_choice
                new_configuration[final_orb_idx] += n_particles_choice

                if (duplicate_configuration := _check_configuration_duplicate(new_configuration=new_configuration, existing_configurations=partition.configurations)):
                    """
                    Check that the newly generated configuration does
                    not already exist.
                    """
                    vum.addstr(
                        vum.n_rows - 3 - vum.command_log_length, 0,
                        f"DUPLICATE: {new_configuration = }, {duplicate_configuration = }"
                    )
                    vum.addstr(
                        vum.n_rows - 2 - vum.command_log_length, 0,
                        f"{partition.configurations[-1].configuration = }"
                    )
                    draw_shell_map(vum=vum, model_space=interaction.model_space.orbitals, is_proton=is_proton, is_neutron=is_neutron)

                    for i in range(len(new_configuration)):
                        draw_shell_map(
                            vum = vum,
                            model_space = interaction.model_space.orbitals,
                            is_proton = is_proton,
                            is_neutron = is_neutron,
                            occupation = (i, new_configuration[i]),
                        )
                    draw_shell_map(
                        vum = vum,
                        model_space = interaction.model_space.orbitals,
                        is_proton = is_proton,
                        is_neutron = is_neutron,
                        occupation = (init_orb_idx, new_configuration[init_orb_idx] + n_particles_choice),
                        n_holes = n_particles_choice,
                    )
                    input_wrapper("Enter any char to continue")

                parity_tmp = _calculate_configuration_parity(
                    configuration = new_configuration,
                    model_space = model_space_slice.orbitals
                )
                if   parity_tmp == -1: partition.n_new_negative_configurations += 1
                elif parity_tmp == +1: partition.n_new_positive_configurations += 1

                ho_quanta_tmp = sum([   # The number of harmonic oscillator quanta for each configuration.
                    n*orb.ho_quanta for n, orb in zip(new_configuration, model_space_slice.orbitals)
                ])
                new_configurations.append(
                    Configuration(
                        configuration = new_configuration.copy(),
                        parity = parity_tmp,
                        ho_quanta = ho_quanta_tmp
                    )
                )
                # draw_shell_map(vum=vum, model_space=interaction.model_space.orbitals, is_proton=is_proton, is_neutron=is_neutron)

                # for i in range(len(new_configuration)):
                #     draw_shell_map(
                #         vum = vum,
                #         model_space = interaction.model_space.orbitals,
                #         is_proton = is_proton,
                #         is_neutron = is_neutron,
                #         occupation = (i, new_configuration[i]),
                #     )
                # draw_shell_map(
                #     vum = vum,
                #     model_space = interaction.model_space.orbitals,
                #     is_proton = is_proton,
                #     is_neutron = is_neutron,
                #     occupation = (init_orb_idx, new_configuration[init_orb_idx] + n_particles_choice),
                #     n_holes = n_particles_choice,
                # )
                # time.sleep(0.5)

    # configurations_formatted += new_configurations    # This can be at the very end, but is added here to be included in the duplicate check later.
    # new_configurations.clear()
    
    for configuration in partition.configurations:
        """
        NOTE: Currently hard-coded for 2 particle excitation. This loop
        deals with the case where 1 particle from 2 different initial
        orbitals are excited into the final major shell.
        """
        # for init_orb_idx in initial_orbital_indices:
        for i1 in range(len(initial_orbital_indices)):
            init_orb_idx_1 = initial_orbital_indices[i1]

            if configuration.configuration[init_orb_idx_1] < 1: continue   # Cannot excite enough particles.

            for i2 in range(i1+1, len(initial_orbital_indices)):
                init_orb_idx_2 = initial_orbital_indices[i2]

                if configuration.configuration[init_orb_idx_2] < 1: continue   # Cannot excite enough particles.


                for f1 in range(len(final_orbital_indices)):
                    final_orb_idx_1 = final_orbital_indices[f1]

                    max_additional_occupation_1 = final_orbital_degeneracy[final_orb_idx_1] - configuration.configuration[final_orb_idx_1]
                    assert max_additional_occupation_1 >= 0, "'max_additional_occupation' should never be negative!"  # Development sanity test.
                    
                    if max_additional_occupation_1 < 1: continue # Cannot excite enough particles.

                    for f2 in range(f1+1, len(final_orbital_indices)):
                        final_orb_idx_2 = final_orbital_indices[f2]

                        max_additional_occupation_2 = final_orbital_degeneracy[final_orb_idx_2] - configuration.configuration[final_orb_idx_2]
                        assert max_additional_occupation_2 >= 0, "'max_additional_occupation' should never be negative!"  # Development sanity test.
                        
                        if max_additional_occupation_2 < 1: continue # Cannot excite enough particles.

                        new_configuration = configuration.configuration.copy()
                        new_configuration[init_orb_idx_1]  -= 1
                        new_configuration[init_orb_idx_2]  -= 1
                        new_configuration[final_orb_idx_1] += 1
                        new_configuration[final_orb_idx_2] += 1

                        if (duplicate_configuration := _check_configuration_duplicate(new_configuration=new_configuration, existing_configurations=partition.configurations)):
                            """
                            Check that the newly generated configuration does
                            not already exist.
                            """
                            vum.addstr(
                                vum.n_rows - 3 - vum.command_log_length, 0,
                                f"DUPLICATE: {new_configuration = }, {duplicate_configuration = }"
                            )
                            vum.addstr(
                                vum.n_rows - 2 - vum.command_log_length, 0,
                                f"{partition.configurations[-1] = }"
                            )
                            draw_shell_map(vum=vum, model_space=interaction.model_space.orbitals, is_proton=is_proton, is_neutron=is_neutron)

                            for i in range(len(new_configuration)):
                                draw_shell_map(
                                    vum = vum,
                                    model_space = interaction.model_space.orbitals,
                                    is_proton = is_proton,
                                    is_neutron = is_neutron,
                                    occupation = (i, new_configuration[i]),
                                )
                            draw_shell_map(
                                vum = vum,
                                model_space = interaction.model_space.orbitals,
                                is_proton = is_proton,
                                is_neutron = is_neutron,
                                occupation = (init_orb_idx_1, new_configuration[init_orb_idx_1] + 1),
                                n_holes = 1,
                            )
                            draw_shell_map(
                                vum = vum,
                                model_space = interaction.model_space.orbitals,
                                is_proton = is_proton,
                                is_neutron = is_neutron,
                                occupation = (init_orb_idx_2, new_configuration[init_orb_idx_2] + 1),
                                n_holes = 1,
                            )
                            input_wrapper("Enter any char to continue")

                        parity_tmp = _calculate_configuration_parity(
                            configuration = new_configuration,
                            model_space = model_space_slice.orbitals
                        )
                        if   parity_tmp == -1: partition.n_new_negative_configurations += 1
                        elif parity_tmp == +1: partition.n_new_positive_configurations += 1

                        ho_quanta_tmp = sum([   # The number of harmonic oscillator quanta for each configuration.
                            n*orb.ho_quanta for n, orb in zip(new_configuration, model_space_slice.orbitals)
                        ])
                        new_configurations.append(
                            Configuration(
                                configuration = new_configuration.copy(),
                                parity = parity_tmp,
                                ho_quanta = ho_quanta_tmp,
                            )
                        )
                        # draw_shell_map(vum=vum, model_space=interaction.model_space.orbitals, is_proton=is_proton, is_neutron=is_neutron)

                        # for i in range(len(new_configuration)):
                        #     draw_shell_map(
                        #         vum = vum,
                        #         model_space = interaction.model_space.orbitals,
                        #         is_proton = is_proton,
                        #         is_neutron = is_neutron,
                        #         occupation = (i, new_configuration[i]),
                        #     )
                        # draw_shell_map(
                        #     vum = vum,
                        #     model_space = interaction.model_space.orbitals,
                        #     is_proton = is_proton,
                        #     is_neutron = is_neutron,
                        #     occupation = (init_orb_idx_1, new_configuration[init_orb_idx_1] + 1),
                        #     n_holes = 1,
                        # )
                        # draw_shell_map(
                        #     vum = vum,
                        #     model_space = interaction.model_space.orbitals,
                        #     is_proton = is_proton,
                        #     is_neutron = is_neutron,
                        #     occupation = (init_orb_idx_2, new_configuration[init_orb_idx_2] + 1),
                        #     n_holes = 1,
                        # )
                        # time.sleep(0.5)

                for f1 in range(len(final_orbital_indices)):
                    """
                    Case: Both particles are excited to the same final
                    orbital.
                    """
                    final_orb_idx_1 = final_orbital_indices[f1]

                    max_additional_occupation_1 = final_orbital_degeneracy[final_orb_idx_1] - configuration.configuration[final_orb_idx_1]
                    assert max_additional_occupation_1 >= 0, "'max_additional_occupation' should never be negative!"  # Development sanity test.
                    
                    if max_additional_occupation_1 < n_particles_choice: continue # Cannot excite enough particles.

                    new_configuration = configuration.configuration.copy()
                    new_configuration[init_orb_idx_1]  -= 1
                    new_configuration[init_orb_idx_2]  -= 1
                    new_configuration[final_orb_idx_1] += n_particles_choice

                    if _check_configuration_duplicate(new_configuration=new_configuration, existing_configurations=new_configurations):
                        """
                        Check for duplicates to the new
                        configurations which were created earlier in
                        this function. There is some overlap so
                        we'll just skip the duplicates here.
                        """
                        continue

                    if (duplicate_configuration := _check_configuration_duplicate(new_configuration=new_configuration, existing_configurations=partition.configurations)):
                        """
                        Check that the newly generated configuration does
                        not already exist.
                        """
                        vum.addstr(
                            vum.n_rows - 3 - vum.command_log_length, 0,
                            f"DUPLICATE: {new_configuration = }, {duplicate_configuration = }"
                        )
                        vum.addstr(
                            vum.n_rows - 2 - vum.command_log_length, 0,
                            f"{partition.configurations[-1] = }"
                        )
                        draw_shell_map(vum=vum, model_space=interaction.model_space.orbitals, is_proton=is_proton, is_neutron=is_neutron)

                        for i in range(len(new_configuration)):
                            draw_shell_map(
                                vum = vum,
                                model_space = interaction.model_space.orbitals,
                                is_proton = is_proton,
                                is_neutron = is_neutron,
                                occupation = (i, new_configuration[i]),
                            )
                        draw_shell_map(
                            vum = vum,
                            model_space = interaction.model_space.orbitals,
                            is_proton = is_proton,
                            is_neutron = is_neutron,
                            occupation = (init_orb_idx_1, new_configuration[init_orb_idx_1] + 1),
                            n_holes = 1,
                        )
                        draw_shell_map(
                            vum = vum,
                            model_space = interaction.model_space.orbitals,
                            is_proton = is_proton,
                            is_neutron = is_neutron,
                            occupation = (init_orb_idx_2, new_configuration[init_orb_idx_2] + 1),
                            n_holes = 1,
                        )
                        input_wrapper("Enter any char to continue")

                    parity_tmp = _calculate_configuration_parity(
                        configuration = new_configuration,
                        model_space = model_space_slice.orbitals
                    )
                    if   parity_tmp == -1: partition.n_new_negative_configurations += 1
                    elif parity_tmp == +1: partition.n_new_positive_configurations += 1

                    ho_quanta_tmp = sum([   # The number of harmonic oscillator quanta for each configuration.
                        n*orb.ho_quanta for n, orb in zip(new_configuration, model_space_slice.orbitals)
                    ])
                    new_configurations.append(
                        Configuration(
                            configuration = new_configuration.copy(),
                            parity = parity_tmp,
                            ho_quanta = ho_quanta_tmp
                        )
                    )
                    # draw_shell_map(vum=vum, model_space=interaction.model_space.orbitals, is_proton=is_proton, is_neutron=is_neutron)

                    # for i in range(len(new_configuration)):
                    #     draw_shell_map(
                    #         vum = vum,
                    #         model_space = interaction.model_space.orbitals,
                    #         is_proton = is_proton,
                    #         is_neutron = is_neutron,
                    #         occupation = (i, new_configuration[i]),
                    #     )
                    # draw_shell_map(
                    #     vum = vum,
                    #     model_space = interaction.model_space.orbitals,
                    #     is_proton = is_proton,
                    #     is_neutron = is_neutron,
                    #     occupation = (init_orb_idx_1, new_configuration[init_orb_idx_1] + 1),
                    #     n_holes = 1,
                    # )
                    # draw_shell_map(
                    #     vum = vum,
                    #     model_space = interaction.model_space.orbitals,
                    #     is_proton = is_proton,
                    #     is_neutron = is_neutron,
                    #     occupation = (init_orb_idx_2, new_configuration[init_orb_idx_2] + 1),
                    #     n_holes = 1,
                    # )
                    # time.sleep(0.5)

    partition.configurations.extend(new_configurations)
    return True

def _sanity_checks(
    partition_proton: Partition,
    partition_neutron: Partition,
    partition_combined: Partition,
    interaction: Interaction,
):
    """
    A few different sanity checks to make sure that the new
    configurations are physical.
    """
    for configuration in partition_proton.configurations:
        assert len(configuration.configuration) == interaction.model_space_proton.n_orbitals
        assert sum(configuration.configuration) == interaction.model_space_proton.n_valence_nucleons
        assert sum([n*orb.ho_quanta for n, orb in zip(configuration.configuration, interaction.model_space_proton.orbitals)]) == configuration.ho_quanta

    for configuration in partition_neutron.configurations:
        assert len(configuration.configuration) == interaction.model_space_neutron.n_orbitals
        assert sum(configuration.configuration) == interaction.model_space_neutron.n_valence_nucleons
        assert sum([n*orb.ho_quanta for n, orb in zip(configuration.configuration, interaction.model_space_neutron.orbitals)]) == configuration.ho_quanta

    for configuration in partition_combined.configurations:
        p_idx, n_idx = configuration.configuration
        
        assert partition_combined.parity == partition_proton.configurations[p_idx].parity*partition_neutron.configurations[n_idx].parity
        assert configuration.ho_quanta == partition_proton.configurations[p_idx].ho_quanta + partition_neutron.configurations[n_idx].ho_quanta

def _prompt_user_for_interaction_and_partition(vum: Vum):
    filenames_interaction = sorted([i for i in os.listdir() if i.endswith(".snt")])
    filenames_partition = sorted([i for i in os.listdir() if i.endswith(".ptn")])
    
    if not filenames_interaction:
        return f"No interaction file present in {os.getcwd()}. Exiting..."
    if not filenames_partition:
        return f"No partition file present in {os.getcwd()}. Exiting..."

    if len(filenames_interaction) == 1:
        filename_interaction = filenames_interaction[0]
        vum.screen.addstr(0, 0, f"{filename_interaction} chosen")
        vum.screen.refresh()

    elif len(filenames_interaction) > 1:
        interaction_choices: str = ""
        for i in range(len(filenames_interaction)):
            interaction_choices += f"{filenames_interaction[i]} ({i}), "
        
        vum.screen.addstr(vum.n_rows - 1 - vum.command_log_length - 1, 0, "Several interaction files detected.")
        vum.screen.addstr(vum.n_rows - 1 - vum.command_log_length, 0, interaction_choices)
        vum.screen.refresh()
        
        while True:
            ans = vum.input("Several interaction files detected. Please make a choice")
            try:
                ans = int(ans)
            except ValueError:
                continue
            
            try:
                filename_interaction = filenames_interaction[ans]
            except IndexError:
                continue

            break

    vum.screen.addstr(0, 0, vum.blank_line)
    vum.screen.addstr(1, 0, vum.blank_line)
    vum.screen.refresh()
    
    if len(filenames_partition) == 1:
        filename_partition = filenames_partition[0]
        vum.screen.addstr(0, 0, f"{filename_partition} chosen")
        vum.screen.refresh()

    elif len(filenames_partition) > 1:
        partition_choices: str = ""
        for i in range(len(filenames_partition)):
            partition_choices += f"{filenames_partition[i]} ({i}), "
        
        vum.screen.addstr(vum.n_rows - 1 - vum.command_log_length - 1, 0, "Several partition files detected.")
        vum.screen.addstr(vum.n_rows - 1 - vum.command_log_length, 0, partition_choices)
        vum.screen.refresh()
        
        while True:
            ans = vum.input("Several partition files detected. Please make a choice")
            try:
                ans = int(ans)
            except ValueError:
                continue
            
            try:
                filename_partition = filenames_partition[ans]
            except IndexError:
                continue

            break
    
    vum.screen.addstr(vum.n_rows - 1 - vum.command_log_length - 1, 0, vum.blank_line)
    vum.screen.addstr(vum.n_rows - 1 - vum.command_log_length, 0, vum.blank_line)
    vum.screen.refresh()

    return filename_interaction, filename_partition

def _load_interaction(
    filename_interaction: str,
    interaction: Interaction
):
    with open(filename_interaction, "r") as infile:
        """
        Extract information from the interaction file about the orbitals
        in the model space.
        """
        for line in infile:
            """
            Example:
            ! 2015/07/13
            ! from GCLSTsdpfsdgix5pn.snt 
            !  8 0g9/2  -0.8MeV
            !  9 0g7/2  -3.2MeV
            ! 10 1d5/2  -2.5MeV
            ! 11 1d3/2  -5.7MeV
            ! 12 2s1/2  -4.0MeV
            !
            ! sdpfsdg_64.sps
            ! sdpfsdg-shell
            ! index,    n,  l,  j, tz
            12  12   8   8
            ...
            """
            if line[0] != "!":
                tmp = line.split()
                interaction.model_space_proton.n_orbitals = int(tmp[0])
                interaction.model_space_neutron.n_orbitals = int(tmp[1])
                interaction.model_space.n_orbitals = (
                    interaction.model_space_proton.n_orbitals + interaction.model_space_neutron.n_orbitals
                )
                interaction.n_core_protons = int(tmp[2])
                interaction.n_core_neutrons = int(tmp[3])
                break

        for line in infile:
            if line[0] == "!": break
            idx, n, l, j, tz = [int(i) for i in line.split("!")[0].split()]
            idx -= 1
            nucleon = "p" if tz == -1 else "n"
            name = f"{n}{spectroscopic_conversion[l]}{j}"
            tmp_orbital = OrbitalParameters(
                idx = idx,
                n = n,
                l = l,
                j = j,
                tz = tz,
                nucleon = nucleon,
                name = f"{nucleon}{name}",
                parity = (-1)**l,
                order = shell_model_order[name],
                ho_quanta = 2*n + l
            )
            interaction.model_space.orbitals.append(tmp_orbital)
            interaction.model_space.major_shell_names.add(shell_model_order[name].major_shell_name)

            if tz == -1:
                interaction.model_space_proton.orbitals.append(tmp_orbital)
                interaction.model_space_proton.major_shell_names.add(shell_model_order[name].major_shell_name)
            elif tz == +1:
                interaction.model_space_neutron.orbitals.append(tmp_orbital)
                interaction.model_space_neutron.major_shell_names.add(shell_model_order[name].major_shell_name)
            else:
                msg = f"Valid values for tz are -1 and +1, got {tz=}"
                raise ValueError(msg)

    interaction.model_space.n_major_shells = len(interaction.model_space.major_shell_names)
    interaction.model_space_proton.n_major_shells = len(interaction.model_space_proton.major_shell_names)
    interaction.model_space_neutron.n_major_shells = len(interaction.model_space_neutron.major_shell_names)

    if not all(orb.idx == i for i, orb in enumerate(interaction.model_space.orbitals)):
        """
        Make sure that the list indices are the same as the orbit
        indices.
        """
        msg = (
            "The orbitals in the model space are not indexed correctly!"
        )
        raise KshellDataStructureError(msg)

def _load_partition(
    filename_partition: str,
    interaction: Interaction,
    partition_proton: Partition,
    partition_neutron: Partition,
    partition_combined: Partition,
) -> str:
    header: str = ""
    with open(filename_partition, "r") as infile:
        # truncation_info: str = infile.readline()    # Eg. hw trucnation,  min hw = 0 ,   max hw = 1
        # hw_min, hw_max = [int(i.split("=")[1].strip()) for i in truncation_info.split(",")[1:]] # NOTE: No idea what happens if no hw trunc is specified.
        for line in infile:
            """
            Extract the information from the header before partitions
            are specified. Example:

            # hw trucnation,  min hw = 0 ,   max hw = 1
            # partition file of gs8.snt  Z=20  N=31  parity=-1
            20 31 -1
            # num. of  proton partition, neutron partition
            86 4
            # proton partition
            ...
            """
            if "#" not in line:
                tmp = [int(i) for i in line.split()]

                try:
                    """
                    For example:
                    20 31 -1
                    """
                    n_valence_protons, n_valence_neutrons, parity_partition = tmp

                    partition_proton.parity = parity_partition
                    partition_neutron.parity = parity_partition
                    partition_combined.parity = parity_partition
                    interaction.model_space.n_valence_nucleons = n_valence_protons + n_valence_neutrons
                    interaction.model_space_proton.n_valence_nucleons = n_valence_protons
                    interaction.model_space_neutron.n_valence_nucleons = n_valence_neutrons
                except ValueError:
                    """
                    For example:
                    86 4
                    """
                    n_proton_configurations, n_neutron_configurations = tmp
                    # partition_proton.n_existing_configurations = n_proton_configurations
                    # partition_neutron.n_existing_configurations = n_neutron_configurations
                    infile.readline()   # Skip header.
                    break

            header += line

        ho_quanta_min: int = +1000  # The actual number of harmonic oscillator quanta will never be smaller or larger than these values.
        ho_quanta_max: int = -1000
        for line in infile:
            """
            Extract proton configurations.
            """
            if "# neutron partition" in line: break

            # proton_configurations.append(line)
            configuration = [int(i) for i in line.split()[1:]]

            parity_tmp = _calculate_configuration_parity(
                configuration = configuration,
                model_space = interaction.model_space_proton.orbitals
            )
            if   parity_tmp == -1: partition_proton.n_existing_negative_configurations += 1
            elif parity_tmp == +1: partition_proton.n_existing_positive_configurations += 1

            assert len(interaction.model_space_proton.orbitals) == len(configuration)

            ho_quanta_tmp = sum([   # The number of harmonic oscillator quanta for each configuration.
                n*orb.ho_quanta for n, orb in zip(configuration, interaction.model_space_proton.orbitals)
            ])
            ho_quanta_min = min(ho_quanta_min, ho_quanta_tmp)
            ho_quanta_max = max(ho_quanta_max, ho_quanta_tmp)
            
            # proton_configurations_formatted.append(
            partition_proton.configurations.append(
                Configuration(
                    configuration = configuration,
                    parity = parity_tmp,
                    ho_quanta = ho_quanta_tmp,
                )
            )
        partition_proton.ho_quanta_min = ho_quanta_min
        partition_proton.ho_quanta_max = ho_quanta_max
        ho_quanta_min: int = +1000  # Reset for neutrons.
        ho_quanta_max: int = -1000
        
        for line in infile:
            """
            Extract neutron configurations.
            """
            if "# partition of proton and neutron" in line: break

            # neutron_configurations.append(line)
            configuration = [int(i) for i in line.split()[1:]]

            parity_tmp = _calculate_configuration_parity(
                configuration = configuration,
                model_space = interaction.model_space_neutron.orbitals
            )
            if   parity_tmp == -1: partition_neutron.n_existing_negative_configurations += 1
            elif parity_tmp == +1: partition_neutron.n_existing_positive_configurations += 1

            assert len(interaction.model_space_neutron.orbitals) == len(configuration)

            ho_quanta_tmp = sum([   # The number of harmonic oscillator quanta for each configuration.
                n*orb.ho_quanta for n, orb in zip(configuration, interaction.model_space_neutron.orbitals)
            ])
            ho_quanta_min = min(ho_quanta_min, ho_quanta_tmp)
            ho_quanta_max = max(ho_quanta_max, ho_quanta_tmp)
            
            # neutron_configurations_formatted.append(
            partition_neutron.configurations.append(
                Configuration(
                    configuration = configuration,
                    parity = parity_tmp,
                    ho_quanta = ho_quanta_tmp,
                )
            )
        partition_neutron.ho_quanta_min = ho_quanta_min
        partition_neutron.ho_quanta_max = ho_quanta_max
        ho_quanta_min: int = +1000  # Reset for combined.
        ho_quanta_max: int = -1000
        n_combined_configurations = int(infile.readline())
        # partition_combined.n_existing_configurations = int(infile.readline())

        for line in infile:
            """
            Extract the combined pn configurations.
            """
            proton_idx, neutron_idx = line.split()
            proton_idx = int(proton_idx) - 1
            neutron_idx = int(neutron_idx) - 1
            # total_configurations_formatted_initial.append([proton_idx, neutron_idx])
            parity_tmp = partition_proton.configurations[proton_idx].parity*partition_neutron.configurations[neutron_idx].parity
            assert parity_partition == parity_tmp

            if   parity_tmp == -1: partition_combined.n_existing_negative_configurations += 1
            elif parity_tmp == +1: partition_combined.n_existing_positive_configurations += 1

            ho_quanta_tmp = (
                partition_proton.configurations[proton_idx].ho_quanta + 
                partition_neutron.configurations[neutron_idx].ho_quanta
            )
            ho_quanta_min = min(ho_quanta_min, ho_quanta_tmp)
            ho_quanta_max = max(ho_quanta_max, ho_quanta_tmp)

            partition_combined.configurations.append(
                Configuration(
                    configuration = [proton_idx, neutron_idx],
                    parity = parity_partition,
                    ho_quanta = ho_quanta_tmp,
                )
            )
        partition_combined.ho_quanta_min = ho_quanta_min
        partition_combined.ho_quanta_max = ho_quanta_max
    
    assert len(partition_proton.configurations) == n_proton_configurations
    assert len(partition_neutron.configurations) == n_neutron_configurations
    assert len(partition_combined.configurations) == n_combined_configurations

    assert (
        partition_proton.n_existing_negative_configurations + partition_proton.n_existing_positive_configurations + 
        partition_proton.n_new_negative_configurations + partition_proton.n_new_positive_configurations
    ) == n_proton_configurations
    assert (
        partition_neutron.n_existing_negative_configurations + partition_neutron.n_existing_positive_configurations + 
        partition_neutron.n_new_negative_configurations + partition_neutron.n_new_positive_configurations
    ) == n_neutron_configurations
    assert (
        partition_combined.n_existing_negative_configurations + partition_combined.n_existing_positive_configurations + 
        partition_combined.n_new_negative_configurations + partition_combined.n_new_positive_configurations
    ) == n_combined_configurations

    return header

def partition_editor(
    filename_partition_edited: str | None = None,
    input_wrapper: Callable | None = None,
):  
    """
    Wrapper for error handling.
    """
    try:
        msg = _partition_editor(
            filename_partition_edited = filename_partition_edited,
        )
    except KeyboardInterrupt:
        curses.endwin()
        print("Exiting without saving changes...")

    except curses.error as e:
        curses.endwin()
        raise(e)
    
    except Exception as e:
        curses.endwin()
        raise(e)

    else:
        curses.endwin()
        print(msg)

def test_partition_editor(
    filename_partition_edited = "test_partition.ptn",
    filename_partition = "Sc44_GCLSTsdpfsdgix5pn_p.ptn",
):
    try:
        os.remove(filename_partition_edited)
    except FileNotFoundError:
        pass

    _partition_editor(
        filename_interaction = "GCLSTsdpfsdgix5pn.snt",
        filename_partition = filename_partition,
        filename_partition_edited = filename_partition_edited,
        vum_wrapper = VumDummy,
    )
    with open(filename_partition_edited, "r") as infile_edited, open(filename_partition, "r") as infile:
        for line_edited, line in zip(infile_edited, infile):
            if line_edited != line:
                line_edited = line_edited.rstrip('\n')
                line = line.rstrip('\n')
                msg = (
                    f"{line_edited} != {line}"
                )
                raise KshellDataStructureError(msg)

def _partition_editor(
    filename_interaction: str | None = None,
    filename_partition: str | None = None,
    filename_partition_edited: str | None = None,
    input_wrapper: Callable | None = None,
    vum_wrapper: Vum = Vum,
):
    """
    Extract the model space orbitals from an interaction file. Extract
    proton and neutron partitions from an accompanying partition file.
    Prompt the user for new proton and neutron configurations and 
    regenerate the proton-neutron partition based on the existing and
    new proton and neutron configurations.

    Here, 'configuration' refers to a specific occupation of protons or
    neutrons in the model space orbitals while 'partition' refers to the
    set of all the configurations. Hence, a 'proton configuration' is
    one specific configuration, like
    ```
    1     6  4  2  0  2  4  2  0  0  0  0  0
    ```
    while the 'proton partition' is the set of all the proton
    configurations, like
    ```
    1     6  4  2  0  2  4  2  0  0  0  0  0
    2     6  4  2  0  3  3  2  0  0  0  0  0
    3     6  4  2  0  3  4  1  0  0  0  0  0
    ...
    ```

    Parameters
    ----------
    filename_partition_edited : str | None
        The name of the edited partition file. If None, a name will be
        generated automatically.

    input_wrapper : Callable
        NOTE: CURRENTLY NOT FUNCTIONING FOR TESTS. Defaults to `input`
        which asks the user for input. This wrapper exists so that unit
        tests can be performed in which case
        `input_wrapper = input_wrapper_test`.
    """
    
    # vum = Vum()
    vum: Vum = vum_wrapper()
    screen = vum.screen
    if input_wrapper is None:
        input_wrapper = vum.input

    screen.clear()  # Clear the screen between interactive sessions.

    if (filename_interaction is None) or (filename_partition is None):
        filename_interaction, filename_partition = \
            _prompt_user_for_interaction_and_partition(vum=vum)

    y_offset: int = 0
    partition_proton: Partition = Partition(
        parity = 0,
        configurations = [],
        n_existing_positive_configurations = 0,
        n_existing_negative_configurations = 0,
        n_new_positive_configurations = 0,
        n_new_negative_configurations = 0,
        ho_quanta_min = 0,
        ho_quanta_max = 0,
    )
    partition_neutron: Partition = Partition(
        parity = 0,
        configurations = [],
        n_existing_positive_configurations = 0,
        n_existing_negative_configurations = 0,
        n_new_positive_configurations = 0,
        n_new_negative_configurations = 0,
        ho_quanta_min = 0,
        ho_quanta_max = 0,
    )
    partition_combined: Partition = Partition(
        parity = 0,
        configurations = [],
        n_existing_positive_configurations = 0,
        n_existing_negative_configurations = 0,
        n_new_positive_configurations = 0,
        n_new_negative_configurations = 0,
        ho_quanta_min = 0,
        ho_quanta_max = 0,
    )
    interaction: Interaction = Interaction(
        model_space = ModelSpace(
            orbitals = [],
            n_major_shells = 0,
            major_shell_names = set(),
            n_orbitals = 0,
            n_valence_nucleons = 0,
        ),
        model_space_proton = ModelSpace(
            orbitals = [],
            n_major_shells = 0,
            major_shell_names = set(),
            n_orbitals = 0,
            n_valence_nucleons = 0,
        ),
        model_space_neutron = ModelSpace(
            orbitals = [],
            n_major_shells = 0,
            major_shell_names = set(),
            n_orbitals = 0,
            n_valence_nucleons = 0,
        ),
        name = filename_interaction,
        n_core_protons = 0,
        n_core_neutrons = 0,
    )
    
    if filename_partition_edited is None:
        filename_partition_edited = f"{filename_partition.split('.')[0]}_edited.ptn"
    
    _load_interaction(filename_interaction=filename_interaction, interaction=interaction)
    draw_shell_map(vum=vum, model_space=interaction.model_space.orbitals, is_proton=True, is_neutron=True)
    header = _load_partition(
        filename_partition = filename_partition,
        interaction = interaction,
        partition_proton = partition_proton,
        partition_neutron = partition_neutron,
        partition_combined = partition_combined,
    )
    combined_existing_configurations_expected: list[Configuration] = partition_combined.configurations.copy()    # For sanity check.
    _generate_total_configurations(
        partition_proton = partition_proton,
        partition_neutron = partition_neutron,
        partition_combined = partition_combined,
        partition_file_parity = partition_combined.parity
    )
    msg = (
        "The number of combined configurations is not correct!"
        f" Expected: {len(combined_existing_configurations_expected)}, calculated: {partition_combined.n_configurations}"
    )
    # return partition_combined.ho_quanta_min, partition_combined.ho_quanta_max
    # assert len(combined_existing_configurations_expected) == partition_combined.n_configurations, msg

    for i in range(partition_combined.n_configurations):
        if partition_combined.configurations[i].configuration != combined_existing_configurations_expected[i].configuration:
            p_idx_expected = combined_existing_configurations_expected[i].configuration[0]
            n_idx_expected = combined_existing_configurations_expected[i].configuration[1]
            p_idx_calculated = partition_combined.configurations[i].configuration[0]
            n_idx_calculated = partition_combined.configurations[i].configuration[1]
            msg = (
                f"Combined config with index {i} is incorrectly calculated!"
                f" Expected: {p_idx_expected, n_idx_expected}, calculated: {p_idx_calculated, n_idx_calculated}"
                f"\nproton_configurations_formatted [{p_idx_expected:5d}] = {partition_proton.configurations[p_idx_expected].configuration}, ho = {partition_proton.configurations[p_idx_expected].ho_quanta}, parity = {partition_proton.configurations[p_idx_expected].parity}"
                f"\nneutron_configurations_formatted[{n_idx_expected:5d}] = {partition_neutron.configurations[n_idx_expected].configuration}, ho = {partition_neutron.configurations[n_idx_expected].ho_quanta}, parity = {partition_neutron.configurations[n_idx_expected].parity}"
                f"\nproton_configurations_formatted [{p_idx_calculated:5d}] = {partition_proton.configurations[p_idx_calculated].configuration}, ho = {partition_proton.configurations[p_idx_calculated].ho_quanta}, parity = {partition_proton.configurations[p_idx_calculated].parity}"
                f"\nneutron_configurations_formatted[{n_idx_calculated:5d}] = {partition_neutron.configurations[n_idx_calculated].configuration}, ho = {partition_neutron.configurations[n_idx_calculated].ho_quanta}, parity = {partition_neutron.configurations[n_idx_calculated].parity}"
                f"\nho expected = {partition_proton.configurations[p_idx_expected].ho_quanta + partition_neutron.configurations[n_idx_expected].ho_quanta}"
                f"\nho calculated = {partition_proton.configurations[p_idx_calculated].ho_quanta + partition_neutron.configurations[n_idx_calculated].ho_quanta}"
                f"\nho min: {partition_combined.ho_quanta_min}"
                f"\nho max: {partition_combined.ho_quanta_max}"
            )
            if partition_combined.configurations[i].configuration not in [c.configuration for c in combined_existing_configurations_expected]:
                msg += f"\n{partition_combined.configurations[i].configuration} should never appear!"
            
            raise KshellDataStructureError(msg)
    
    # idx = np.argmin([c.ho_quanta for c in partition_proton.configurations])
    # print(partition_proton.configurations[idx])
    # print([c.ho_quanta for c in interaction.model_space_proton.orbitals])
    # return

    if isinstance(vum, VumDummy):
        """
        Skip dim calculation for testing.
        """
        M = [0]
        mdim = [0]
    else:
        M, mdim, jdim = count_dim(
            model_space_filename = filename_interaction,
            partition_filename = None,
            print_dimensions = False,
            debug = False,
            parity = partition_combined.parity,
            proton_partition = [configuration.configuration for configuration in partition_proton.configurations],
            neutron_partition = [configuration.configuration for configuration in partition_neutron.configurations],
            total_partition = [configuration.configuration for configuration in partition_combined.configurations],
        )
    mdim_original: int = mdim[-1]
    vum.addstr(y_offset, 0, f"{filename_interaction}, {filename_partition}")
    vum.addstr(y_offset + 1, 0, f"M-scheme dim (M={M[-1]}): {mdim[-1]:d} ({mdim[-1]:.2e})")
    vum.addstr(y_offset + 2, 0, f"n proton, neutron configurations: {partition_proton.n_configurations}, {partition_neutron.n_configurations}")
    vum.addstr(y_offset + 3, 0, f"n valence protons, neutrons: {interaction.model_space_proton.n_valence_nucleons}, {interaction.model_space_neutron.n_valence_nucleons}")
    vum.addstr(y_offset + 4, 0, f"n core protons, neutrons: {interaction.n_core_protons}, {interaction.n_core_neutrons}")
    vum.addstr(y_offset + 5, 0, f"{partition_combined.parity = }")
    vum.addstr(y_offset + 6, 0, f"n proton +, - : {partition_proton.n_existing_positive_configurations}, {partition_proton.n_existing_negative_configurations}")
    vum.addstr(y_offset + 7, 0, f"n neutron +, - : {partition_neutron.n_existing_positive_configurations}, {partition_neutron.n_existing_negative_configurations}")
    vum.addstr(y_offset + 8, 0, "parity current configuration = None")
    vum.addstr(y_offset + 9, 0, f"n pn configuration combinations: {partition_combined.n_configurations}")

    _analyse_existing_configuration(
        vum = vum,
        proton_configurations = partition_proton.configurations,
        neutron_configurations = partition_neutron.configurations,
        input_wrapper = input_wrapper,
        model_space = interaction.model_space.orbitals,
        y_offset = y_offset,
        n_proton_orbitals = interaction.model_space_proton.n_orbitals,
    )
    while True:
        configuration_choice = input_wrapper("Add configuration? (y/n)")
        if configuration_choice == "y":
            pass
        elif configuration_choice in ["n", "q"]:
            break
        else: continue

        while True:
            nucleon_choice = input_wrapper("Proton or neutron? (p/n)")
            if nucleon_choice == "p":
                is_proton = True
                is_neutron = False
                nucleon = "proton"
                model_space_slice = interaction.model_space_proton
                partition = partition_proton
                n_valence_nucleons = interaction.model_space_proton.n_valence_nucleons
                break
            
            elif nucleon_choice == "n":
                is_proton = False
                is_neutron = True
                nucleon = "neutron"
                model_space_slice = interaction.model_space_neutron
                partition = partition_neutron
                n_valence_nucleons = interaction.model_space_neutron.n_valence_nucleons
                break

            else: continue

        while True:
            draw_shell_map(vum=vum, model_space=interaction.model_space.orbitals, is_proton=is_proton, is_neutron=is_neutron)
            configuration_type_choice = input_wrapper("Single or range of configurations? (s/r)")
            if configuration_type_choice == "s":
                """
                Prompt the user for single specific configurations.
                """
                while True:
                    new_configuration = _prompt_user_for_configuration(
                        vum = vum,
                        nucleon = nucleon,
                        model_space = model_space_slice.orbitals,
                        n_valence_nucleons = n_valence_nucleons,
                        input_wrapper = input_wrapper,
                        y_offset = y_offset,
                    )
                    if new_configuration:
                        if _check_configuration_duplicate(
                            new_configuration = new_configuration,
                            existing_configurations = partition.configurations
                            ):
                                msg = (
                                    "This configuration already exists! Skipping..."
                                )
                                vum.addstr(vum.n_rows - 1 - vum.command_log_length - 2, 0, "DUPLICATE")
                                vum.addstr(vum.n_rows - 1 - vum.command_log_length - 1, 0, msg)
                                time.sleep(DELAY)
                                draw_shell_map(vum=vum, model_space=interaction.model_space.orbitals, is_proton=is_proton, is_neutron=is_neutron)
                                continue

                        parity_tmp = _calculate_configuration_parity(
                            configuration = new_configuration,
                            model_space = model_space_slice.orbitals
                        )
                        if   parity_tmp == -1: partition.n_new_negative_configurations += 1
                        elif parity_tmp == +1: partition.n_new_positive_configurations += 1

                        ho_quanta_tmp = sum([   # The number of harmonic oscillator quanta for each configuration.
                            n*orb.ho_quanta for n, orb in zip(new_configuration, interaction.model_space_proton.orbitals)
                        ])
                        partition.configurations.append(
                            Configuration(
                                configuration = new_configuration,
                                parity = parity_tmp,
                                ho_quanta = ho_quanta_tmp,
                            )
                        )
                        try:
                            _generate_total_configurations(
                                partition_proton = partition_proton,
                                partition_neutron = partition_neutron,
                                partition_combined = partition_combined,
                                partition_file_parity = partition_combined.parity
                            )
                        except KshellDataStructureError:
                            """
                            Accept invalid configuration parity for now
                            since the user might specify more
                            configurations which may multiplicatively
                            combine parities to match the .ptn parity.
                            """
                            pass

                        M, mdim, jdim = count_dim(
                            model_space_filename = filename_interaction,
                            partition_filename = None,
                            print_dimensions = False,
                            debug = False,
                            parity = partition_combined.parity,
                            proton_partition = [configuration.configuration for configuration in partition_proton.configurations],
                            neutron_partition = [configuration.configuration for configuration in partition_neutron.configurations],
                            total_partition = [configuration.configuration for configuration in partition_combined.configurations],
                        )
                        vum.addstr(y_offset + 1, 0, f"M-scheme dim (M={M[-1]}): {mdim[-1]:d} ({mdim[-1]:.2e}) (original {mdim_original:d} ({mdim_original:.2e}))")
                        vum.addstr(y_offset + 2, 0, f"n proton, neutron configurations: {partition_proton.n_existing_configurations} + {partition_proton.n_new_configurations}, {partition_neutron.n_existing_configurations} + {partition_neutron.n_new_configurations}")
                        vum.addstr(y_offset + 6, 0, f"n proton +, - : {partition_proton.n_existing_positive_configurations} + {partition_proton.n_new_positive_configurations}, {partition_proton.n_existing_negative_configurations} + {partition_proton.n_new_negative_configurations}")
                        vum.addstr(y_offset + 7, 0, f"n neutron +, - : {partition_neutron.n_existing_positive_configurations} + {partition_neutron.n_new_positive_configurations}, {partition_neutron.n_existing_negative_configurations} + {partition_neutron.n_new_negative_configurations}")
                        vum.addstr(y_offset + 9, 0, f"n pn configuration combinations: {partition_combined.n_configurations}")
                        vum.addstr(vum.n_rows - 1 - vum.command_log_length - 2, 0, "Configuration added!")
                        time.sleep(DELAY)

                    elif new_configuration is None:
                        """
                        Quit signal. Do not keep the current configuration,
                        but keep earlier defined new configurations and quit
                        the prompt.
                        """
                        draw_shell_map(vum=vum, model_space=interaction.model_space.orbitals, is_proton=is_proton, is_neutron=is_neutron)
                        vum.addstr(vum.n_rows - 3 - vum.command_log_length, 0, " ")
                        # vum.addstr(vum.n_rows - 2 - vum.command_log_length, 0, "Current configuration discarded")
                        vum.addstr(vum.n_rows - 2 - vum.command_log_length, 0, " ")
                        break
                break

            elif configuration_type_choice == "r":
                """
                Prompt the user for a range of configurations. Currently
                only supports 2p2h excitations.
                """
                if not _add_npnh_excitations(
                    vum = vum,
                    input_wrapper = input_wrapper,
                    model_space_slice = model_space_slice,
                    interaction = interaction,
                    partition = partition,
                    nucleon_choice = nucleon_choice,
                    is_proton = is_proton,
                    is_neutron = is_neutron
                ):
                    continue
                try:
                    _generate_total_configurations(
                        partition_proton = partition_proton,
                        partition_neutron = partition_neutron,
                        partition_combined = partition_combined,
                        partition_file_parity = partition_combined.parity
                    )
                except KshellDataStructureError:
                    """
                    Allow parity mismatch for now.
                    """
                    pass

                M, mdim, jdim = count_dim(
                    model_space_filename = filename_interaction,
                    partition_filename = None,
                    print_dimensions = False,
                    debug = False,
                    parity = partition_combined.parity,
                    proton_partition = [configuration.configuration for configuration in partition_proton.configurations],
                    neutron_partition = [configuration.configuration for configuration in partition_neutron.configurations],
                    total_partition = [configuration.configuration for configuration in partition_combined.configurations],
                )
                vum.addstr(y_offset + 1, 0, f"M-scheme dim (M={M[-1]}): {mdim[-1]:d} ({mdim[-1]:.2e}) (original {mdim_original:d} ({mdim_original:.2e}))")
                vum.addstr(y_offset + 2, 0, f"n proton, neutron configurations: {partition_proton.n_existing_configurations} + {partition_proton.n_new_configurations}, {partition_neutron.n_existing_configurations} + {partition_neutron.n_new_configurations}")
                vum.addstr(y_offset + 6, 0, f"n proton +, - : {partition_proton.n_existing_positive_configurations} + {partition_proton.n_new_positive_configurations}, {partition_proton.n_existing_negative_configurations} + {partition_proton.n_new_negative_configurations}")
                vum.addstr(y_offset + 7, 0, f"n neutron +, - : {partition_neutron.n_existing_positive_configurations} + {partition_neutron.n_new_positive_configurations}, {partition_neutron.n_existing_negative_configurations} + {partition_neutron.n_new_negative_configurations}")
                vum.addstr(y_offset + 9, 0, f"n pn configuration combinations: {partition_combined.n_configurations}")
                vum.addstr(vum.n_rows - 1 - vum.command_log_length - 2, 0, "Configurations added!")
                break

            elif configuration_type_choice == "q":
                break

            else: continue

    n_new_proton_configurations = partition_proton.n_new_negative_configurations + partition_proton.n_new_positive_configurations
    n_new_neutron_configurations = partition_neutron.n_new_negative_configurations + partition_neutron.n_new_positive_configurations
    
    # if (not n_new_neutron_configurations) and (not n_new_proton_configurations):
    #     return "No new configurations. Skipping creation of new .ptn file."

    _generate_total_configurations(
        partition_proton = partition_proton,
        partition_neutron = partition_neutron,
        partition_combined = partition_combined,
        partition_file_parity = partition_combined.parity
    )
    _sanity_checks(
        partition_proton = partition_proton,
        partition_neutron = partition_neutron,
        partition_combined = partition_combined,
        interaction = interaction,
    )
    with open(filename_partition_edited, "w") as outfile:
        """
        Write edited data to new partition file.
        """
        outfile.write(header)
        outfile.write(f" {partition_proton.n_configurations} {partition_neutron.n_configurations}\n")
        outfile.write("# proton partition\n")
        
        # for configuration in proton_configurations:
        #     outfile.write(configuration)

        # for i in range(n_new_proton_configurations):
        #     outfile.write(
        #         f"{i + partition_proton.n_configurations + 1:6d}     "    # +1 because .ptn indices start at 1.
        #         f"{str(new_proton_configurations[i]).strip('[]').replace(',', ' ')}"
        #         "\n"
        #     )

        for i in range(partition_proton.n_configurations):
            outfile.write(
                f"{i + 1:6d}     "    # +1 because .ptn indices start at 1.
                f"{str(partition_proton.configurations[i].configuration).strip('[]').replace(',', ' ')}"
                "\n"
            )
        outfile.write("# neutron partition\n")

        # for configuration in neutron_configurations:
        #     outfile.write(configuration)

        # for i in range(n_new_neutron_configurations):
        #     outfile.write(
        #         f"{i + partition_neutron.n_configurations + 1:6d}     "    # +1 because .ptn indices start at 1.
        #         f"{str(new_neutron_configurations[i]).strip('[]').replace(',', ' ')}"
        #         "\n"
        #     )
        for i in range(partition_neutron.n_configurations):
            outfile.write(
                f"{i + 1:6d}     "    # +1 because .ptn indices start at 1.
                f"{str(partition_neutron.configurations[i].configuration).strip('[]').replace(',', ' ')}"
                "\n"
            )
        outfile.write("# partition of proton and neutron\n")
        outfile.write(f"{partition_combined.n_configurations}\n")

        # for configuration in total_configurations_formatted:
        #     outfile.write(f"{configuration[0] + 1:5d} {configuration[1] + 1:5d}\n") # +1 because .ptn indices start at 1.

        for configuration in partition_combined.configurations:
            outfile.write(f"{configuration.configuration[0] + 1:5d} {configuration.configuration[1] + 1:5d}\n") # +1 because .ptn indices start at 1.

    return (
        f"New configuration saved to {filename_partition_edited}"
        f" with {n_new_proton_configurations} new proton configurations"
        f" and {n_new_neutron_configurations} new neutron configurations"
    )

def _prompt_user_for_configuration(
    vum: Vum,
    nucleon: str,
    model_space: list[OrbitalParameters],
    n_valence_nucleons: int,
    input_wrapper: Callable,
    y_offset: int,
) -> list | None:
        """
        Prompt the user for a new proton or neutron configuration,
        meaning that this function prompts the user for the occupation
        of all the orbitals in the model space and returns the
        configuration as a list of occupation numbers. Note that the
        order of the orbitals that the user is prompted for is sorted
        by the `shell_model_order` dictionary but the occupation number
        list returned by this function is sorted by the order of the
        orbitals as defined in the interaction file.

        Parameters
        ----------
        vum : Vum
            The Vum object to use for displaying information.

        nucleon : str
            The nucleon type to prompt the user for. Must be either
            "proton" or "neutron".
        
        model_space : list[OrbitalParameters]
            The model space to use for the prompt.

        n_valence_nucleons : int
            The number of valence protons or neutrons. Used for making
            sure that the user does not enter more than this number of
            nucleons in the configuration.

        input_wrapper : Callable
            The input wrapper to use for getting user input.

        y_offset : int
            The y-offset to use for displaying information.

        Returns
        -------
        list | None
            The occupation of each model space orbitals as a list of
            occupation numbers. If the user enters "q" to quit, None is
            returned.
        """
        is_blank_map: bool = True
        if nucleon == "proton":
            is_proton = True
            is_neutron = False

        elif nucleon == "neutron":
            is_proton = False
            is_neutron = True

        else:
            msg = f"Invalid nucleon parameter '{nucleon}'!"
            raise ValueError(msg)

        model_space_copy = sorted(  # Sort the orbitals based on the shell_model_order dict.
            model_space,
            key = lambda orbital: shell_model_order[f"{orbital.n}{spectroscopic_conversion[orbital.l]}{orbital.j}"].idx,
        )
        n_remaining_nucleons: int = n_valence_nucleons
        vum.addstr(vum.n_rows - 3 - vum.command_log_length, 0, " ")
        vum.addstr(vum.n_rows - 2 - vum.command_log_length, 0, f"Please enter {nucleon} orbital occupation (f to fill, q to quit):")
        occupation: list[tuple[int, int]] = []
        occupation_parity: int = 1

        for orbital in model_space_copy:
            if n_remaining_nucleons == 0:
                """
                If there are no more valence nucleons to use, set the
                remaining occupations to 0.
                """
                occupation.append((orbital.idx, 0))
                vum.addstr(vum.n_rows - 3 - vum.command_log_length, 0, "Occupation of remaining orbitals set to 0.")
                continue

            while True:
                ans = input_wrapper(f"{orbital.idx + 1:2d} {orbital} (remaining: {n_remaining_nucleons})")
                if is_blank_map:
                    is_blank_map = False
                    draw_shell_map(
                        vum = vum,
                        model_space = model_space,
                        is_proton = is_proton,
                        is_neutron = is_neutron,
                    )

                if (ans == "q") or (ans == "quit") or (ans == "exit"): return None
                
                if ans == "f":
                    """
                    Fill the orbital completely or with as many
                    remaining nucleons as possible.
                    """
                    ans = min(orbital.j + 1, n_remaining_nucleons)
                
                try:
                    ans = int(ans)
                except ValueError:
                    continue

                if (ans > (orbital.j + 1)) or (ans < 0):
                    vum.addstr(
                        vum.n_rows - 3 - vum.command_log_length, 0,
                        f"Allowed occupation for this orbital is [0, 1, ..., {orbital.j + 1}]"
                    )
                    continue
                
                occupation_parity *= orbital.parity**ans
                vum.addstr(y_offset + 8, 0, f"parity current configuration = {occupation_parity}")
                
                n_remaining_nucleons -= ans
                if ans > 0:
                    """
                    No point in re-drawing the line if the occupation is
                    unchanged.
                    """
                    draw_shell_map(
                        vum = vum,
                        model_space = model_space,
                        is_proton = is_proton,
                        is_neutron = is_neutron,
                        occupation = (orbital.idx, ans)
                    )
                break

            occupation.append((orbital.idx, ans))

            cum_occupation = sum(tup[1] for tup in occupation)
            if cum_occupation > n_valence_nucleons:
                vum.addstr(
                    vum.n_rows - 3 - vum.command_log_length, 0,
                    f"INVALID: Total occupation ({cum_occupation}) exceeds the number of valence {nucleon}s ({n_valence_nucleons})"
                )
                draw_shell_map(
                    vum = vum,
                    model_space = model_space,
                    is_proton = is_proton,
                    is_neutron = is_neutron,
                )
                return []
        
        cum_occupation = sum(tup[1] for tup in occupation)
        if cum_occupation < n_valence_nucleons:
            vum.addstr(
                vum.n_rows - 3 - vum.command_log_length, 0,
                f"INVALID: Total occupation ({cum_occupation}) does not use the total number of valence {nucleon}s ({n_valence_nucleons})"
            )
            draw_shell_map(
                vum = vum,
                model_space = model_space,
                is_proton = is_proton,
                is_neutron = is_neutron,
            )
            return []
        
        vum.addstr(vum.n_rows - 2 - vum.command_log_length, 0, vum.blank_line)
        occupation.sort(key=lambda tup: tup[0]) # Sort list of tuples based on the orbital.idx.
        return [tup[1] for tup in occupation]   # Return only the occupation numbers now sorted based on the orbital order of the .snt file.