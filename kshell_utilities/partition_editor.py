from __future__ import annotations
import time, os, curses, warnings
from math import inf
from typing import Callable
from vum import Vum
from .count_dim import count_dim
from .kshell_exceptions import KshellDataStructureError
from .parameters import (
    spectroscopic_conversion, shell_model_order, major_shell_order
)
from .data_structures import (
    OrbitalParameters, Configuration, ModelSpace, Interaction, Partition, Skips
)
from .partition_tools import (
    _prompt_user_for_interaction_and_partition, _calculate_configuration_parity,
    _sanity_checks, configuration_energy,
)
from .loaders import load_partition, load_interaction
from .partition_compare import _partition_compare

warnings.filterwarnings("ignore", category=UserWarning)
DELAY: int = 2  # Delay time for time.sleep(DELAY) in seconds
PARITY_CURRENT_Y_COORD = 5
# is_duplicate_warning = True
y_offset: int = 0

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
        self.blank_line = ""

    def addstr(self, y, x, string, is_blank_line=None):
        return

    def input(self, _) -> str:
        return "n"
    
class VumDummy2(VumDummy):
    def __init__(self) -> None:
        super().__init__()
        # self.answers: list[str] = [
        #     "n",    # Analyse?
        #     "y",    # Add config?
        #     "p",    # Proton or neutron?
        #     "r",    # Single or range?
        #     "2",    # N particle N hole?
        #     "2",    # How many excitations to add? NOTE: Currently not in use.
        #     "pf",   # Initial major shell?
        #     "sdg",  # Final major shell?
        #     "n",    # Add config?
        # ]
        self.answers: list[str] = [
            "n",
            "y",
            "p",
            "s",
            "f",
            "f",
            "2",
            "2",
            "2",
            "2",
            "2",
            "2",
        ]
    
    def input(self, _) -> str:
        ans = self.answers.pop(0)
        print(ans)
        return ans

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

                    vum.addstr(y_offset + PARITY_CURRENT_Y_COORD, 0, f"parity current configuration = {configuration_parity}")
                
                else:
                    draw_shell_map(vum=vum, model_space=model_space, is_proton=is_proton, is_neutron=is_neutron)

        else: break # If answer from user is not 'y'.

    vum.addstr(y_offset + PARITY_CURRENT_Y_COORD, 0, "parity current configuration = None")
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
    y_offset: int = 14
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

def _summary_information(
    vum: Vum,
    skips: Skips,
    filename_interaction: str,
    filename_partition: str,
    partition_proton: Partition,
    partition_neutron: Partition,
    partition_combined: Partition,
    interaction: Interaction,
    M: list[int],
    mdim: list[int],
    # n_proton_skips: int,
    # n_neutron_skips: int,
    # n_parity_skips: int,
    # n_ho_skips: int,
    y_offset: int,
    mdim_original: int | None = None,
):
    vum.addstr(
        y = y_offset,
        x = 0,
        string = f"{filename_interaction} -- {filename_partition} -- parity: {partition_combined.parity} -- core pn: ({interaction.n_core_protons}, {interaction.n_core_neutrons}) -- valence pn: ({interaction.model_space_proton.n_valence_nucleons}, {interaction.model_space_neutron.n_valence_nucleons})"
    )
    
    if mdim_original is None:
        vum.addstr(
            y = y_offset + 1,
            x = 0,
            string = f"M-scheme dim (M={M[-1]}): {mdim[-1]:d} ({mdim[-1]:.2e})"
        )
    else:
        vum.addstr(
            y = y_offset + 1,
            x = 0,
            string = f"M-scheme dim (M={M[-1]}): {mdim[-1]:d} ({mdim[-1]:.2e}) (original {mdim_original:d} ({mdim_original:.2e}))"
        )
    # vum.addstr(
    #     y = y_offset + 2,
    #     x = 0,
    #     string = f"n valence protons, neutrons: {interaction.model_space_proton.n_valence_nucleons}, {interaction.model_space_neutron.n_valence_nucleons}"
    # )
    # vum.addstr(
    #     y = y_offset + 3,
    #     x = 0,
    #     string = f"n core protons, neutrons: {interaction.n_core_protons}, {interaction.n_core_neutrons}"
    # )
    vum.addstr(
        y = y_offset + 2,
        x = 0,
        string = f"n proton +, -, sum : ({partition_proton.n_existing_positive_configurations} + {partition_proton.n_new_positive_configurations}), ({partition_proton.n_existing_negative_configurations} + {partition_proton.n_new_negative_configurations}), ({partition_proton.n_existing_configurations} + {partition_proton.n_new_configurations})",
    )
    vum.addstr(
        y = y_offset + 3,
        x = 0,
        string = f"n neutron +, -, sum : ({partition_neutron.n_existing_positive_configurations} + {partition_neutron.n_new_positive_configurations}), ({partition_neutron.n_existing_negative_configurations} + {partition_neutron.n_new_negative_configurations}), ({partition_neutron.n_existing_configurations} + {partition_neutron.n_new_configurations})"
    )
    vum.addstr(
        y = y_offset + 4,
        x = 0,
        string = f"n combined: {partition_combined.n_configurations}"
    )
    vum.addstr(
        y = y_offset + PARITY_CURRENT_Y_COORD,
        x = 0,
        string = "parity current configuration = None"
    )
    vum.addstr(
        y = y_offset + 6,
        x = 0,
        string = f"n proton, neutron configurations will be deleted because of parity or H.O. mismatch: {skips.n_proton_skips, skips.n_neutron_skips}"
    )
    vum.addstr(
        y = y_offset + 7,
        x = 0,
        string = f"n parity, H.O. skips: {skips.n_parity_skips, skips.n_ho_skips}"
    )
    vum.addstr(
        y = y_offset + 8,
        x = 0,
        string = f"Monopole trunc skips: {skips.n_monopole_skips}"
    )
    if partition_combined.max_configuration_energy_original == partition_combined.max_configuration_energy:
        vum.addstr(
            y = y_offset + 9,
            x = 0,
            string = f"Min, max, diff configuration energy: {partition_combined.min_configuration_energy:.2f}, {partition_combined.max_configuration_energy:.2f}, {abs(partition_combined.min_configuration_energy - partition_combined.max_configuration_energy):.2f}"
        )
    else:
        vum.addstr(
            y = y_offset + 9,
            x = 0,
            string = f"Min, max, diff configuration energy: {partition_combined.min_configuration_energy:.2f}, {partition_combined.max_configuration_energy:.2f} (original {partition_combined.max_configuration_energy_original:.2f}), {abs(partition_combined.min_configuration_energy - partition_combined.max_configuration_energy):.2f}"
        )
    vum.addstr(
        y = y_offset + 10,
        x = 0,
        string = f"Min H.O.: {partition_combined.ho_quanta_min}, max H.O.: {partition_combined.ho_quanta_max}"
    )

def _generate_total_configurations(
    interaction: Interaction,
    partition_proton: Partition,
    partition_neutron: Partition,
    partition_combined: Partition,
    partition_file_parity: int,
    skips: Skips,
    threshold_energy: float,
    allow_invalid: bool = True,
    is_recursive: bool = False,
):# -> tuple[int, int]:
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
    ho_quanta_min_before = partition_combined.ho_quanta_min_this_parity
    ho_quanta_max_before = partition_combined.ho_quanta_max_this_parity
    assert partition_combined.ho_quanta_min != +1000    # Check that they're not at the default values.
    assert partition_combined.ho_quanta_max != -1000

    partition_combined.clear()
    neutron_configurations_count: list[int] = [0]*partition_neutron.n_configurations
    neutron_configurations_parity_skips: list[int] = [0]*partition_neutron.n_configurations
    neutron_configurations_ho_skips: list[int] = [0]*partition_neutron.n_configurations
    proton_configurations_count: list[int] = [0]*partition_proton.n_configurations
    proton_configurations_parity_skips: list[int] = [0]*partition_proton.n_configurations
    proton_configurations_ho_skips: list[int] = [0]*partition_proton.n_configurations
    n_monopole_skips = 0
    
    for p_idx in range(partition_proton.n_configurations):
        for n_idx in range(partition_neutron.n_configurations):
            parity_tmp: int = partition_proton.configurations[p_idx].parity*partition_neutron.configurations[n_idx].parity
            if parity_tmp != partition_file_parity:
                """
                Only combinations of proton and neutron orbitals with
                the same parity as the parity of the partition file are
                accepted.
                """
                neutron_configurations_parity_skips[n_idx] += 1
                proton_configurations_parity_skips[p_idx] += 1
                continue

            ho_quanta_tmp: int = (
                partition_proton.configurations[p_idx].ho_quanta + 
                partition_neutron.configurations[n_idx].ho_quanta
            )
            if not (partition_combined.ho_quanta_min <= ho_quanta_tmp <= partition_combined.ho_quanta_max):
                """
                The combined harmonic oscillator quanta of the combined
                proton and neutron orbitals must be within the initial
                limits.
                """
                neutron_configurations_ho_skips[n_idx] += 1
                proton_configurations_ho_skips[p_idx] += 1
                continue
            
            is_original = partition_proton.configurations[p_idx].is_original and partition_neutron.configurations[n_idx].is_original
            energy = configuration_energy(
                interaction = interaction,
                proton_configuration = partition_proton.configurations[p_idx],
                neutron_configuration = partition_neutron.configurations[n_idx],
            )
            if (energy > threshold_energy):# and not is_original:
                """
                Skip configurations with energies over the threshold
                energy only if they are newly generated configurations
                and not originally existing configurations.
                """
                n_monopole_skips += 1
                continue
            
            if energy < partition_combined.min_configuration_energy:
                """
                I am not yet sure if this is OK or not.
                """
                msg = (
                    "A newly calculated pn configuration energy is lower than"
                    " the initial lowest energy pn configuration!"
                )
                raise RuntimeError(msg)

            neutron_configurations_count[n_idx] += 1
            proton_configurations_count[p_idx] += 1

            if partition_file_parity == -1: partition_combined.n_new_negative_configurations += 1
            if partition_file_parity == +1: partition_combined.n_new_positive_configurations += 1

            partition_combined.ho_quanta_min_this_parity = min(partition_combined.ho_quanta_min_this_parity, ho_quanta_tmp)
            partition_combined.ho_quanta_max_this_parity = max(partition_combined.ho_quanta_max_this_parity, ho_quanta_tmp)

            partition_combined.configurations.append(
                Configuration(
                    configuration = [p_idx, n_idx],
                    parity = parity_tmp,
                    ho_quanta = ho_quanta_tmp,
                    energy = energy,
                    is_original = is_original,
                )
            )
    if is_recursive:
        assert len(proton_configurations_count) == len(partition_proton.configurations)
        assert len(neutron_configurations_count) == len(partition_neutron.configurations)
        assert 0 not in proton_configurations_count, proton_configurations_count # Every config should be used now.
        assert 0 not in neutron_configurations_count, neutron_configurations_count
        assert (
            partition_proton.n_existing_positive_configurations +
            partition_proton.n_existing_negative_configurations +
            partition_proton.n_new_positive_configurations +
            partition_proton.n_new_negative_configurations
        ) == partition_proton.n_configurations
        assert (
            partition_neutron.n_existing_positive_configurations +
            partition_neutron.n_existing_negative_configurations +
            partition_neutron.n_new_positive_configurations +
            partition_neutron.n_new_negative_configurations
        ) == partition_neutron.n_configurations
        assert (
            partition_combined.n_existing_positive_configurations +
            partition_combined.n_existing_negative_configurations +
            partition_combined.n_new_positive_configurations +
            partition_combined.n_new_negative_configurations
        ) == partition_combined.n_configurations
        
        return # The recursive call should only re-generate combined configs.
    
    if not allow_invalid:
        """
        Remove proton and neutron configurations which have not been
        counted.
        """
        for p_idx in reversed(range(partition_proton.n_configurations)):
            """
            Traverse configurations in reverse to preserve the order of
            deletion. Consider the list

                ['0', '1', '2', '3', '4'].

            If we are to delete elements 1 and 3, starting from index 0,
            we would first delete '1' at position 1 and then '4' at
            position 3. In reverse however, we would delete '3' at
            position 3 and then '1' at position 1.
            """
            if proton_configurations_count[p_idx] == 0:
                if partition_proton.configurations[p_idx].parity == -1:
                    partition_proton.n_new_negative_configurations -= 1

                elif partition_proton.configurations[p_idx].parity == +1:
                    partition_proton.n_new_positive_configurations -= 1

                else:
                    raise KshellDataStructureError
                
                partition_proton.configurations.pop(p_idx)

        for n_idx in reversed(range(partition_neutron.n_configurations)):
            if neutron_configurations_count[n_idx] == 0:
                if partition_neutron.configurations[n_idx].parity == -1:
                    partition_neutron.n_new_negative_configurations -= 1

                elif partition_neutron.configurations[n_idx].parity == +1:
                    partition_neutron.n_new_positive_configurations -= 1

                else:
                    raise KshellDataStructureError
                
                partition_neutron.configurations.pop(n_idx)

        _generate_total_configurations( # Re-generate combined partition after removing invalid p and n configs.
            interaction = interaction,
            partition_proton = partition_proton,
            partition_neutron = partition_neutron,
            partition_combined = partition_combined,
            partition_file_parity = partition_file_parity,
            skips = skips,
            threshold_energy = threshold_energy,
            is_recursive = True,
        )

    # assert ho_quanta_min_before == partition_combined.ho_quanta_min_this_parity, f"{ho_quanta_min_before} != {partition_combined.ho_quanta_min_this_parity}"
    # assert ho_quanta_max_before == partition_combined.ho_quanta_max_this_parity, f"{ho_quanta_max_before} != {partition_combined.ho_quanta_max_this_parity}"
    
    partition_combined.max_configuration_energy = max([configuration.energy for configuration in partition_combined.configurations])
    skips.n_proton_skips = len([i for i in proton_configurations_count if i == 0])    # The number of proton configurations which will be removed at the end of this program.
    skips.n_neutron_skips = len([i for i in neutron_configurations_count if i == 0])  # The number of neutron configurations which will be removed at the end of this program.
    skips.n_parity_skips = sum(proton_configurations_parity_skips) + sum(neutron_configurations_parity_skips)
    skips.n_ho_skips = sum(proton_configurations_ho_skips) + sum(neutron_configurations_ho_skips)
    skips.n_monopole_skips = n_monopole_skips

def _check_configuration_duplicate(
    new_configuration: list[int],
    existing_configurations: list[Configuration],
    vum: Vum,
    interaction: Interaction,
    is_proton: bool,
    is_neutron: bool,
    init_orb_idx: int,
    n_particles_choice: int,
    is_duplicate_warning: bool,
) -> bool:
    """
    allow_duplicate_warning : bool
        The duplicate warnings should just be displayed if the user
        requested configurations have duplicates in the existing
        configurations. This program might have some overlap in the
        configuration generation algorithms and in these cases no
        duplicate warning should be displayed.
    """
    
    for i, configuration in enumerate(existing_configurations):
        if new_configuration == configuration.configuration:
            duplicate_configuration =  [str(i), configuration]
            break
    else:
        return False
    
    if not is_duplicate_warning: return True
    
    vum.addstr(
        vum.n_rows - 3 - vum.command_log_length, 0,
        f"DUPLICATE: {new_configuration = }, {duplicate_configuration = }"
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
        
    return True

def _add_npnh_excitations(
    vum: Vum,
    input_wrapper: Callable,
    model_space_slice: ModelSpace,
    interaction: Interaction,
    partition: Partition,
    partition_proton: Partition,
    partition_neutron: Partition,
    partition_combined: Partition,
    nucleon_choice: str,
    is_proton: bool,
    is_neutron: bool,
) -> bool:
    while True:
        """
        Prompt user for the number of particles to excite.
        """
        n_particles_choice = input_wrapper("N-particle N-hole (N)")
        if n_particles_choice == "q": return False
        
        try:
            n_particles_choice = int(n_particles_choice)
        except ValueError:
            continue

        if n_particles_choice > 2:
            vum.addstr(
                vum.n_rows - 3 - vum.command_log_length, 0,
                "INVALID: Only supports 1 and 2 particle excitations!"
            )
            continue

        if n_particles_choice < 1:
            vum.addstr(
                vum.n_rows - 3 - vum.command_log_length, 0,
                "INVALID: The number of particles must be larger than 0."
            )
            continue

        break

    while True:
        """
        Prompt the user for which major shells to include
        in the N-particle N-hole excitation.
        """
        initial_major_shell_choice = input_wrapper(f"Initial major shell? ({model_space_slice.major_shell_names})")
        if initial_major_shell_choice == "q": return False

        if initial_major_shell_choice in model_space_slice.major_shell_names: break

    while True:
        """
        Prompt the user for which major shells to include
        in the N-particle N-hole excitation.
        """
        final_major_shell_choice = input_wrapper(f"Final major shell? ({model_space_slice.major_shell_names})")
        if final_major_shell_choice == "q": return False

        if final_major_shell_choice in model_space_slice.major_shell_names: break
    
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
    is_duplicate_warning: bool = True
    n_duplicate_skips: int = 0

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

    while True:
        remove_initial_orbital_choice = input_wrapper(f"Remove any of the initial orbitals? ({[idx + 1 for idx in initial_orbital_indices]}/n)")
        if remove_initial_orbital_choice == "n": break
        if remove_initial_orbital_choice == "q": return

        try:
            remove_initial_orbital_choice = int(remove_initial_orbital_choice)
        except ValueError:
            continue

        remove_initial_orbital_choice -= 1

        try:
            initial_orbital_indices.remove(remove_initial_orbital_choice)
        except ValueError:
            continue

        if not initial_orbital_indices: return

    while True:
        remove_final_orbital_choice = input_wrapper(f"Remove any of the final orbitals? ({[idx + 1 for idx in final_orbital_indices]}/n)")
        if remove_final_orbital_choice == "n": break
        if remove_final_orbital_choice == "q": return

        try:
            remove_final_orbital_choice = int(remove_final_orbital_choice)
        except ValueError:
            continue

        remove_final_orbital_choice -= 1

        try:
            final_orbital_indices.remove(remove_final_orbital_choice)
        except ValueError:
            continue

        if not final_orbital_indices: return

    if is_proton:
        new_ho_quanta_max = partition_proton.ho_quanta_max_this_parity + partition_neutron.ho_quanta_max_this_parity
        new_ho_quanta_max -= n_particles_choice*interaction.model_space_proton.orbitals[initial_orbital_indices[-1]].ho_quanta
        new_ho_quanta_max += n_particles_choice*interaction.model_space_proton.orbitals[final_orbital_indices[-1]].ho_quanta

    if is_neutron:
        new_ho_quanta_max = partition_proton.ho_quanta_max_this_parity + partition_neutron.ho_quanta_max_this_parity
        new_ho_quanta_max -= n_particles_choice*interaction.model_space_neutron.orbitals[initial_orbital_indices[-1]].ho_quanta
        new_ho_quanta_max += n_particles_choice*interaction.model_space_neutron.orbitals[final_orbital_indices[-1]].ho_quanta

    if new_ho_quanta_max > partition_combined.ho_quanta_max:
        while True:
            ho_quanta_max_choice = input_wrapper(f"This NpNh will exceed the max ho quanta ({partition_combined.ho_quanta_max}) with up to {new_ho_quanta_max - partition_combined.ho_quanta_max}. New max?")
            if ho_quanta_max_choice == "q": return False

            try:
                ho_quanta_max_choice = int(ho_quanta_max_choice)
            except ValueError:
                continue

            if ho_quanta_max_choice < partition_combined.ho_quanta_max:
                vum.addstr(
                    vum.n_rows - 3 - vum.command_log_length, 0,
                    "INVALID: New max H.O. quanta must be larger than the old value!"
                )
                continue

            partition_combined.ho_quanta_max = ho_quanta_max_choice
            break
            
    """
    NOTE: Gjøre slik at jeg kan velge å legge til n NpNh-eksitasjoner og
    at de er de med n lavest energi. `init_orb_idx` og `final_orb_idx`
    inneholder nok det jeg trenger for å gjøre dette valget. Et
    alternativ er å begrense hele orbitaler i stedet for å velge antall
    eksitasjoner. F.eks. å bare helt droppe den orbitalen med høyest
    energi.
    """
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

                is_duplicate_configuration = _check_configuration_duplicate(
                    new_configuration = new_configuration,
                    existing_configurations = new_configurations,
                    vum = vum,
                    interaction = interaction,
                    is_proton = is_proton,
                    is_neutron = is_neutron,
                    init_orb_idx = init_orb_idx,
                    n_particles_choice = n_particles_choice,
                    is_duplicate_warning = False
                )

                if is_duplicate_configuration:
                    """
                    Checking against the newly generated configuraions.
                    """
                    continue
                
                is_duplicate_configuration = _check_configuration_duplicate(
                    new_configuration = new_configuration,
                    existing_configurations = partition.configurations,
                    vum = vum,
                    interaction = interaction,
                    is_proton = is_proton,
                    is_neutron = is_neutron,
                    init_orb_idx = init_orb_idx,
                    n_particles_choice = n_particles_choice,
                    is_duplicate_warning = is_duplicate_warning,
                )
                if is_duplicate_configuration:
                    """
                    Check that the newly generated configuration does
                    not already exist.
                    """
                    n_duplicate_skips += 1
                    if not is_duplicate_warning: continue
                    duplicate_choice = vum.input("Enter any char to continue or 'i' to ignore duplicate warnings (they will still be deleted)")
                    if duplicate_choice == "i": is_duplicate_warning = False
                    continue

                parity_tmp = _calculate_configuration_parity(
                    configuration = new_configuration,
                    model_space = model_space_slice.orbitals,
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
                        energy = None,
                        is_original = False,
                    )
                )

    if n_particles_choice == 1:
        """
        The next loop of this function only applies for the 2 particle
        excitation case.
        """
        partition.configurations.extend(new_configurations)
        if n_duplicate_skips:
            vum.addstr(
                vum.n_rows - 3 - vum.command_log_length, 0,
                f"DUPLICATE SKIPS: {n_duplicate_skips}"
            )
        return True
    
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
                        
                        is_duplicate_configuration = _check_configuration_duplicate(
                            new_configuration = new_configuration,
                            existing_configurations = new_configurations,
                            vum = vum,
                            interaction = interaction,
                            is_proton = is_proton,
                            is_neutron = is_neutron,
                            init_orb_idx = init_orb_idx,
                            n_particles_choice = n_particles_choice,
                            is_duplicate_warning = False
                        )

                        if is_duplicate_configuration:
                            """
                            Checking against the newly generated configuraions.
                            """
                            continue
                        
                        is_duplicate_configuration = _check_configuration_duplicate(
                            new_configuration = new_configuration,
                            existing_configurations = partition.configurations,
                            vum = vum,
                            interaction = interaction,
                            is_proton = is_proton,
                            is_neutron = is_neutron,
                            init_orb_idx = init_orb_idx,
                            n_particles_choice = n_particles_choice,
                            is_duplicate_warning = is_duplicate_warning,
                        )
                        if is_duplicate_configuration:
                            """
                            Check that the newly generated configuration does
                            not already exist.
                            """
                            n_duplicate_skips += 1
                            if not is_duplicate_warning: continue
                            duplicate_choice = vum.input("Enter any char to continue or 'i' to ignore duplicate warnings (they will still be deleted)")
                            if duplicate_choice == "i": is_duplicate_warning = False
                            continue

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
                                energy = None,
                                is_original = False,
                            )
                        )

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

                    is_duplicate_configuration = _check_configuration_duplicate(
                        new_configuration = new_configuration,
                        existing_configurations = new_configurations,
                        vum = vum,
                        interaction = interaction,
                        is_proton = is_proton,
                        is_neutron = is_neutron,
                        init_orb_idx = init_orb_idx,
                        n_particles_choice = n_particles_choice,
                        is_duplicate_warning = False
                    )

                    if is_duplicate_configuration:
                        """
                        Checking against the newly generated configuraions.
                        """
                        continue
                    
                    is_duplicate_configuration = _check_configuration_duplicate(
                        new_configuration = new_configuration,
                        existing_configurations = partition.configurations,
                        vum = vum,
                        interaction = interaction,
                        is_proton = is_proton,
                        is_neutron = is_neutron,
                        init_orb_idx = init_orb_idx,
                        n_particles_choice = n_particles_choice,
                        is_duplicate_warning = is_duplicate_warning,
                    )
                    if is_duplicate_configuration:
                        """
                        Check that the newly generated configuration does
                        not already exist.
                        """
                        n_duplicate_skips += 1
                        if not is_duplicate_warning: continue
                        duplicate_choice = vum.input("Enter any char to continue or 'i' to ignore duplicate warnings (they will still be deleted)")
                        if duplicate_choice == "i": is_duplicate_warning = False
                        continue

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
                            energy = None,
                            is_original = False,
                        )
                    )

    partition.configurations.extend(new_configurations)
    if n_duplicate_skips:
        vum.addstr(
            vum.n_rows - 3 - vum.command_log_length, 0,
            f"DUPLICATE SKIPS: {n_duplicate_skips}"
        )
    return True

def partition_editor(
    filename_partition_edited: str | None = None,
):  
    """
    Wrapper for error handling.
    """
    try:
        vum: Vum = Vum()
        vum.screen.clear()  # Clear the screen between interactive sessions.
        program_choice = vum.input("Edit or compare? (e/c)")
        if program_choice == "e":
            msg = _partition_editor(
                filename_partition_edited = filename_partition_edited,
                vum_wrapper = vum,
            )
        elif program_choice == "c":
            msg = _partition_compare(vum=vum)
        
        else: return

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
    filename_partition: str,
    filename_interaction: str,
    filename_partition_edited: str = "test_partition.ptn",
):
    try:
        os.remove(filename_partition_edited)
    except FileNotFoundError:
        pass
    
    vum: VumDummy = VumDummy()
    _partition_editor(
        filename_interaction = filename_interaction,
        filename_partition = filename_partition,
        filename_partition_edited = filename_partition_edited,
        vum_wrapper = vum,
    )
    n_lines: int = 0
    with open(filename_partition_edited, "r") as infile_edited, open(filename_partition, "r") as infile:
        for line_edited, line in zip(infile_edited, infile):
            n_lines += 1
            if line_edited != line:
                line_edited = line_edited.rstrip('\n')
                line = line.rstrip('\n')
                msg = (
                    f"{line_edited} != {line}"
                )
                raise KshellDataStructureError(msg)
        # else:
        #     print(f"All {n_lines} lines are identical in the files {filename_partition} and {filename_partition_edited}!")

    try:
        os.remove(filename_partition_edited)
    except FileNotFoundError:
        pass

def test_partition_editor_2(
    filename_partition_edited = "test_partition.ptn",
    filename_partition = "Ni67_gs8_n.ptn",
):
    try:
        os.remove(filename_partition_edited)
    except FileNotFoundError:
        pass
    
    vum: VumDummy2 = VumDummy2()
    ans = _partition_editor(
        filename_interaction = "gs8.snt",
        filename_partition = filename_partition,
        filename_partition_edited = filename_partition_edited,
        vum_wrapper = vum,
    )

    try:
        os.remove(filename_partition_edited)
    except FileNotFoundError:
        pass

    print(ans)

def _partition_editor(
    vum_wrapper: Vum,
    filename_interaction: str | None = None,
    filename_partition: str | None = None,
    filename_partition_edited: str | None = None,
    is_recursive = False,
) -> tuple[int, int] | None:
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
    """
    global return_string
    return_string = ""  # For returning a message to be printed after the screen closes.
    
    vum: Vum = vum_wrapper
    screen = vum.screen
    input_wrapper = vum.input

    # screen.clear()  # Clear the screen between interactive sessions.

    if (filename_interaction is None) or (filename_partition is None):
        tmp = _prompt_user_for_interaction_and_partition(vum=vum, is_compare_mode=False)
        if not tmp: return "Exiting without saving changes..."
        filename_interaction, filename_partition = tmp
    
    filename_partition_opposite_parity: str = \
        filename_partition[0:-5] + ("n" if (filename_partition[-5] == "p") else "p") + filename_partition[-4:]

    partition_proton: Partition = Partition()
    partition_neutron: Partition = Partition()
    partition_combined: Partition = Partition()
    interaction: Interaction = Interaction()
    skips: Skips = Skips()
    threshold_energy: float = inf    # Threshold energy for monopole truncation. inf means no truncation.
    
    if os.path.isfile(filename_partition_opposite_parity) and not is_recursive:
        """
        Perform a recursive call to this function to determine the min
        and max ho quanta of the opposite parity partition.
        """
        partition_combined.ho_quanta_min_opposite_parity, partition_combined.ho_quanta_max_opposite_parity = _partition_editor(
            filename_interaction = filename_interaction,
            filename_partition = filename_partition_opposite_parity,
            vum_wrapper = VumDummy(),
            is_recursive = True,
        )
    elif not os.path.isfile(filename_partition_opposite_parity):
        """
        This is not a problem if the calculations only have a single
        parity or if there is no hw truncation.
        """
        vum.addstr(vum.n_rows - 2 - vum.command_log_length, 0, (
            f"WARNING! Could not find opposite parity partition file,"
            " thus cannot properly calculate min and max H.O. quanta "
            " for proper treatment of hw truncation."
        ))
        time.sleep(DELAY)

    if filename_partition_edited is None:
        filename_partition_edited = f"{filename_partition.split('.')[0]}_edited.ptn"
    
    load_interaction(filename_interaction=filename_interaction, interaction=interaction)
    draw_shell_map(vum=vum, model_space=interaction.model_space.orbitals, is_proton=True, is_neutron=True)
    header = load_partition(
        filename_partition = filename_partition,
        interaction = interaction,
        partition_proton = partition_proton,
        partition_neutron = partition_neutron,
        partition_combined = partition_combined,
    )

    if is_recursive:
        """
        This happens only inside the recursive function call.
        """
        return partition_combined.ho_quanta_min_this_parity, partition_combined.ho_quanta_max_this_parity

    combined_existing_configurations_expected: list[Configuration] = partition_combined.configurations.copy()    # For sanity check.

    while True:
        _generate_total_configurations(
            interaction = interaction,
            threshold_energy = threshold_energy,
            partition_proton = partition_proton,
            partition_neutron = partition_neutron,
            partition_combined = partition_combined,
            partition_file_parity = partition_combined.parity,
            skips = skips,
        )
        if len(combined_existing_configurations_expected) != partition_combined.n_configurations:
            msg = (
                "The number of combined configurations is not correct!"
                f" Expected: {len(combined_existing_configurations_expected)}, calculated: {partition_combined.n_configurations}"
            )
            vum.addstr(vum.n_rows - 1 - vum.command_log_length - 2, 0, msg)
            msg = (
                "This can happen if the .ptn file was generated with monopole"
                " truncation."
            )
            vum.addstr(vum.n_rows - 1 - vum.command_log_length - 1, 0, msg)
            while True:
                check_choice = input_wrapper(".ptn monopole threshold energy or skip the check? (energy/s)")
                if check_choice == "s": break

                try:
                    threshold_energy = float(check_choice)
                except ValueError:
                    continue
                else:
                    threshold_energy = partition_combined.min_configuration_energy + threshold_energy
                    break

            if check_choice == "s": break

        else: break

    vum.addstr(vum.n_rows - 1 - vum.command_log_length - 2, 0, vum.blank_line)
    vum.addstr(vum.n_rows - 1 - vum.command_log_length - 1, 0, vum.blank_line)

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
                f"\nho min: {partition_combined.ho_quanta_min_this_parity}"
                f"\nho max: {partition_combined.ho_quanta_max_this_parity}"
            )
            if partition_combined.configurations[i].configuration not in [c.configuration for c in combined_existing_configurations_expected]:
                msg += f"\n{partition_combined.configurations[i].configuration} should never appear!"
            
            raise KshellDataStructureError(msg)

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
    _summary_information(
        vum = vum,
        skips = skips,
        filename_interaction = filename_interaction,
        filename_partition = filename_partition,
        partition_proton = partition_proton,
        partition_neutron = partition_neutron,
        partition_combined = partition_combined,
        interaction = interaction,
        M = M,
        mdim = mdim,
        y_offset = y_offset,
        mdim_original = None,
        # n_proton_skips = 0,
        # n_neutron_skips = 0,
        # n_parity_skips = 0,
        # n_ho_skips = 0,
    )
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

            elif nucleon_choice == "q":
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
                        is_duplicate_configuration = _check_configuration_duplicate(
                            new_configuration = new_configuration,
                            existing_configurations = partition.configurations,
                            vum = vum,
                            interaction = interaction,
                            is_proton = is_proton,
                            is_neutron = is_neutron,
                            init_orb_idx = 0,
                            n_particles_choice = 0,
                            is_duplicate_warning = False,
                        )
                        if is_duplicate_configuration:
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
                        return_string += f"\n{len(partition_proton.configurations) = }, {len(partition_neutron.configurations) = }, {len(partition_combined.configurations) = }"
                        partition.configurations.append(
                            Configuration(
                                configuration = new_configuration,
                                parity = parity_tmp,
                                ho_quanta = ho_quanta_tmp,
                                energy = None,
                                is_original = False,
                            )
                        )
                        return_string += f"\n{len(partition_proton.configurations) = }, {len(partition_neutron.configurations) = }, {len(partition_combined.configurations) = }"
                        _generate_total_configurations(
                            interaction = interaction,
                            threshold_energy = threshold_energy,
                            partition_proton = partition_proton,
                            partition_neutron = partition_neutron,
                            partition_combined = partition_combined,
                            partition_file_parity = partition_combined.parity,
                            allow_invalid = True,
                            skips = skips,
                        )
                        # try:
                        #     n_proton_skips, n_neutron_skips = _generate_total_configurations(
                        #         partition_proton = partition_proton,
                        #         partition_neutron = partition_neutron,
                        #         partition_combined = partition_combined,
                        #         partition_file_parity = partition_combined.parity,
                        #         allow_invalid = True,
                        #     )
                        # except KshellDataStructureError:
                        #     """
                        #     Accept invalid configuration parity for now
                        #     since the user might specify more
                        #     configurations which may multiplicatively
                        #     combine parities to match the .ptn parity.
                        #     """
                        #     pass
                        
                        # print(f"n proton, neutron configurations will be skipped because of parity or H.O. mismatch: {n_proton_skips, n_neutron_skips}")
                        return_string += f"\n{len(partition_proton.configurations) = }, {len(partition_neutron.configurations) = }, {len(partition_combined.configurations) = }"
                        # return return_string
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
                        # vum.addstr(y_offset + 1, 0, f"M-scheme dim (M={M[-1]}): {mdim[-1]:d} ({mdim[-1]:.2e}) (original {mdim_original:d} ({mdim_original:.2e}))")
                        # vum.addstr(y_offset + 2, 0, f"n proton, neutron configurations: {partition_proton.n_existing_configurations} + {partition_proton.n_new_configurations}, {partition_neutron.n_existing_configurations} + {partition_neutron.n_new_configurations}")
                        # vum.addstr(y_offset + 6, 0, f"n proton +, - : {partition_proton.n_existing_positive_configurations} + {partition_proton.n_new_positive_configurations}, {partition_proton.n_existing_negative_configurations} + {partition_proton.n_new_negative_configurations}")
                        # vum.addstr(y_offset + 7, 0, f"n neutron +, - : {partition_neutron.n_existing_positive_configurations} + {partition_neutron.n_new_positive_configurations}, {partition_neutron.n_existing_negative_configurations} + {partition_neutron.n_new_negative_configurations}")
                        # vum.addstr(y_offset + 9, 0, f"n pn configuration combinations: {partition_combined.n_configurations}")
                        _summary_information(
                            vum = vum,
                            filename_interaction = filename_interaction,
                            filename_partition = filename_partition,
                            partition_proton = partition_proton,
                            partition_neutron = partition_neutron,
                            partition_combined = partition_combined,
                            interaction = interaction,
                            M = M,
                            mdim = mdim,
                            y_offset = y_offset,
                            mdim_original = mdim_original,
                            skips = skips,
                            # n_proton_skips = n_proton_skips,
                            # n_neutron_skips = n_neutron_skips,
                            # n_parity_skips = n_parity_skips,
                            # n_ho_skips = n_ho_skips,
                        )
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
                if input_wrapper("Monopole truncation? (y/n)") == "y":
                    while True:
                        try:
                            threshold_energy = input_wrapper("Monopole trunc threshold energy (relative to min)")
                            threshold_energy = float(threshold_energy)

                        except ValueError:
                            continue

                        if threshold_energy > 0:
                            threshold_energy = partition_combined.min_configuration_energy + threshold_energy
                            break

                if not _add_npnh_excitations(
                    vum = vum,
                    input_wrapper = input_wrapper,
                    model_space_slice = model_space_slice,
                    interaction = interaction,
                    partition = partition,
                    partition_proton = partition_proton,
                    partition_neutron = partition_neutron,
                    partition_combined = partition_combined,
                    nucleon_choice = nucleon_choice,
                    is_proton = is_proton,
                    is_neutron = is_neutron,
                ):
                    continue
                try:
                    _generate_total_configurations(
                        interaction = interaction,
                        threshold_energy = threshold_energy,
                        partition_proton = partition_proton,
                        partition_neutron = partition_neutron,
                        partition_combined = partition_combined,
                        partition_file_parity = partition_combined.parity,
                        skips = skips,
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
                # vum.addstr(y_offset + 1, 0, f"M-schemelol dim (M={M[-1]}): {mdim[-1]:d} ({mdim[-1]:.2e}) (original {mdim_original:d} ({mdim_original:.2e}))")
                # vum.addstr(y_offset + 2, 0, f"n proton, neutron configurations: {partition_proton.n_existing_configurations} + {partition_proton.n_new_configurations}, {partition_neutron.n_existing_configurations} + {partition_neutron.n_new_configurations}")
                # vum.addstr(y_offset + 6, 0, f"n proton +, - : {partition_proton.n_existing_positive_configurations} + {partition_proton.n_new_positive_configurations}, {partition_proton.n_existing_negative_configurations} + {partition_proton.n_new_negative_configurations}")
                # vum.addstr(y_offset + 7, 0, f"n neutron +, - : {partition_neutron.n_existing_positive_configurations} + {partition_neutron.n_new_positive_configurations}, {partition_neutron.n_existing_negative_configurations} + {partition_neutron.n_new_negative_configurations}")
                # vum.addstr(y_offset + 9, 0, f"n pn configuration combinations: {partition_combined.n_configurations}")
                _summary_information(
                    vum = vum,
                    filename_interaction = filename_interaction,
                    filename_partition = filename_partition,
                    partition_proton = partition_proton,
                    partition_neutron = partition_neutron,
                    partition_combined = partition_combined,
                    interaction = interaction,
                    M = M,
                    mdim = mdim,
                    y_offset = y_offset,
                    mdim_original = mdim_original,
                    skips = skips,
                    # n_proton_skips = n_proton_skips,
                    # n_neutron_skips = n_neutron_skips,
                    # n_parity_skips = n_parity_skips,
                    # n_ho_skips = n_ho_skips,
                )
                # vum.addstr(vum.n_rows - 1 - vum.command_log_length - 2, 0, "Configurations added!")
                break

            elif configuration_type_choice == "q":
                break

            else: continue

    _generate_total_configurations(
        interaction = interaction,
        partition_proton = partition_proton,
        partition_neutron = partition_neutron,
        partition_combined = partition_combined,
        partition_file_parity = partition_combined.parity,
        threshold_energy = threshold_energy,    # NOTE: Maybe this should be inf to avoid removing existing configurations?
        allow_invalid = False,
        skips = skips,
    )
    n_new_proton_configurations = partition_proton.n_new_negative_configurations + partition_proton.n_new_positive_configurations
    n_new_neutron_configurations = partition_neutron.n_new_negative_configurations + partition_neutron.n_new_positive_configurations

    if not isinstance(vum, VumDummy):
        """
        Dont create a .ptn file if no changes are made, but allow it for
        unit tests using VumDummy.
        """
        if (not n_new_neutron_configurations) and (not n_new_proton_configurations):
            return_string += "No new configurations. Skipping creation of new .ptn file."
            return return_string
    
    _sanity_checks(
        partition_proton = partition_proton,
        partition_neutron = partition_neutron,
        partition_combined = partition_combined,
        interaction = interaction,
    )
    while True:
        save_changes_choice = input_wrapper("Save changes to new partition file? (y/n)")
        
        if save_changes_choice == "n":
            return_string += "Exiting without saving changes..."
            return return_string
        
        elif save_changes_choice == "y": break

    with open(filename_partition_edited, "w") as outfile:
        """
        Write edited data to new partition file.
        """
        outfile.write(header)
        outfile.write(f" {partition_proton.n_configurations} {partition_neutron.n_configurations}\n")
        outfile.write("# proton partition\n")

        for i in range(partition_proton.n_configurations):
            outfile.write(
                f"{i + 1:6d}     "    # +1 because .ptn indices start at 1.
                f"{str(partition_proton.configurations[i].configuration).strip('[]').replace(',', ' ')}"
                "\n"
            )
        outfile.write("# neutron partition\n")

        for i in range(partition_neutron.n_configurations):
            outfile.write(
                f"{i + 1:6d}     "    # +1 because .ptn indices start at 1.
                f"{str(partition_neutron.configurations[i].configuration).strip('[]').replace(',', ' ')}"
                "\n"
            )
        outfile.write("# partition of proton and neutron\n")
        outfile.write(f"{partition_combined.n_configurations}\n")

        for configuration in partition_combined.configurations:
            outfile.write(f"{configuration.configuration[0] + 1:5d} {configuration.configuration[1] + 1:5d}\n") # +1 because .ptn indices start at 1.

    return_string += (
        f"New configuration saved to {filename_partition_edited}"
        f" with {n_new_proton_configurations} new proton configurations"
        f" and {n_new_neutron_configurations} new neutron configurations"
    )
    return return_string

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
                vum.addstr(y_offset + PARITY_CURRENT_Y_COORD, 0, f"parity current configuration = {occupation_parity}")
                
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
